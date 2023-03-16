'''
 - This script performs the grasping of all objects in the environment and
   drops them in the target bin and back to the environment.
 - It uses image feedback from a ceiling rgbd camera and on robot gripper
   camera.
 - It also tries to push object away from the edge to increase the chances
   of a successful grasp.
'''

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array, from_euler_angles
import cv2
import multiprocessing
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from rlbench.backend import task
import gym
from gym import spaces
import matplotlib.pyplot as plt


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def sample_normal_pose(pos_scale, rot_scale):
    '''
    Samples a 6D pose from a zero-mean isotropic normal distribution
    '''
    pos = np.random.normal(scale=pos_scale)

    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)

    return pos, quat_wxyz


def locateObject(depth):
    (height, width) = depth.shape
    depth = ((depth/np.max(depth))*255).astype('uint8')
    center = [depth.shape[1]//2, depth.shape[0]*2//3]

    proc = cv2.adaptiveThreshold(
        depth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(proc, proc)
    for h in range(height):
        cv2.floodFill(thresh, None, (h, 0), 0)
        cv2.floodFill(thresh, None, (h, width - 1), 0)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(
        cv2.erode(thresh, kernel, iterations=1), kernel, iterations=4)

    for h in range(height):
        cv2.floodFill(thresh, None, (h, 0), 0)
        cv2.floodFill(thresh, None, (h, width - 1), 0)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    numCount = len(contours)
    minIdx = -1
    minDist = None
    minCenter = None
    for i in range(numCount):
        c = contours[i]
        area = cv2.contourArea(c)
        if area > width*height/4:
            continue
        cX = int(np.average(c[:, :, 0]))
        cY = int(np.average(c[:, :, 1]))
        distance = sp.spatial.distance.euclidean(center, [cX, cY])
        if minIdx == -1 or minDist > distance:
            minDist = distance
            minIdx = i
            minCenter = (cX, cY)
    if minDist == None:
        return None
    rect = cv2.minAreaRect(contours[minIdx])
    (x, y), (width, height), angle = rect
    if (width < height):
        angle -= 90
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(depth, [box], 0, (0, 0, 255), 2)
    return depth, thresh, minCenter, np.array(minCenter) - np.array(center), angle + 90


def discrete(state):
    t = np.zeros(8)
    t[:7] = np.around(np.rad2deg(state[:7]))
    t[7] = state[7] > 0.5
    return np.int_(t)


class RandomAgent:

    def __init__(self, sourceBin, targetBin, env):
        self._env = env
        self._flag = True
        self.flag = True
        self.position = 0
        self._push = True
        #self._obj_pos = obj_pos
        self._sourcePosition = sourceBin[:3]
        self._sourcePosition[2] -= 0.5
        self._targetPosition = targetBin[:3]
        self.reset()
        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3
        self.policy = {}
        self.q = {}
        self.value = {}
        self.alpha = 0.5
        self.epsilon = 0.2
        self.gamma = 0.95
        self.isResetting = False
        self.action_space = spaces.Box(
            low=np.concatenate([np.ones(7)*(-1), [0]], axis=-1), high=np.concatenate([np.ones(7), [1]], axis=-1), dtype=np.int)

    def reset(self):
        self._graspState = 0
        self._countPull = 0
        self.isRotated = False
        self.isTranslated = False
        self.lastPose = np.zeros(11)

    def act(self, obs):
        current = np.concatenate(
            [obs.joint_positions, [obs.gripper_open]], axis=-1)
        action = self.action_space.sample()
        if np.random.random_sample() > self.epsilon and hash(str(discrete(current))) in self.policy:
            action = self.policy.get(hash(str(discrete(current))), action)

        nextAction = np.zeros(8)
        nextAction[:7] = np.deg2rad(action[:7])
        nextAction[7] = action[7]
        return current, nextAction

    def updateQ(self, state, action, nextState, reward):
        current, _, _ = self.q.get(
            hash(str((discrete(state), action))), (0, None, None))
        #print(str((discrete(state), action, discrete(nextState))))
        self.q[hash(str((discrete(state), action)))] = ((1 - self.alpha)*current + self.alpha*(reward + self.gamma *
                                                                                               self.value.get(hash(str(discrete(nextState))), 0)), state, action)

    def updateV(self):
        for (key, data) in self.q.items():
            (value, state, action) = data
            current = self.value.get(hash(str(discrete(state))), 0)
            if current < value:
                # print(value)
                self.value[hash(str(discrete(state)))] = value
                self.policy[hash(str(discrete(state)))] = action


    def resetAct(self, obs, poses):
        # current = np.concatenate([obs.gripper_pose, [obs.gripper_open , self._graspState, self.isRotated, self.isTranslated]], axis=-1)
        # if np.allclose(np.array(self.lastPose) - current, np.zeros(11), atol=0.001) and self._graspState == 3:
        #     print("Cannot pull back after grasp, release and try again")
        #     res = np.concatenate([obs.gripper_pose, [1]], axis=-1)
        #     self._graspState = 2
        #     return res
        # self.lastPose = current
        print("state", self._graspState)
        if self._graspState == 0:
            # Move to bin
            delta_pos = self._sourcePosition
            delta_quat = R.from_euler('y', 180, degrees=True).as_quat()
            gripper_pos = [1]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
            res[2] = 0.95
            diff = obs.gripper_pose-res[: 7]
            if np.allclose(diff, np.zeros(7), atol=0.05):
                self._graspState += 1
                self.isRotated = False
                self.isTranslated = False
                print("Move close to grasp object now")
        elif self._graspState == 1:
            # Locate object
            depth = np.array(obs.wrist_depth)
            data = locateObject(depth)
            if data == None:
                self.reset()
                self.isResetting = False
                res = np.concatenate([obs.gripper_pose, [1]], axis=-1)
                return res

            depth, thres, center, delta, angle = data
            delta = np.array(delta)/1000
            targetR = R.from_euler('z', angle, degrees=True)
            nextRotationR = R.from_quat(
                obs.gripper_pose[3: 7]).as_matrix()@targetR.as_matrix()
            delta_pos = obs.gripper_pose[: 3]
            
            if not self.isTranslated:
                print("Translating Gripper")
                
                mask = (np.abs(delta) > 0.002)
                delta = delta * mask
                if np.allclose(delta, np.zeros(2), atol=0.0005):
                    mask1 = self._env._scene._cam_wrist_mask.capture_rgb()
                    mask1 = mask1[:, : ,0]*255
                    ret = DetectEdge(center, mask1)
                    if ret != None:
                        dis, direction, point = ret
                        if dis < 100:
                            self._graspState = 10
                    self.isTranslated = True
                else:
                    rotationM = np.eye(4)
                    rotationM[: 3, : 3] = R.from_quat(
                        obs.gripper_pose[3: 7]).as_matrix()
                    rotationM[: 3, 3] = np.array(obs.gripper_pose[: 3])
                    p = np.array([delta[1], -delta[0], 0, 1])
                    p = rotationM@p
                    delta_pos = p[: 3]
                delta_quat = obs.gripper_pose[3: 7]
            elif not self.isRotated:
                delta_quat = obs.gripper_pose[3: 7]
                delta_quat = R.from_matrix(nextRotationR).as_quat()
                self.isRotated = True
                print("Rotating Gripper")
            else:
                self._graspState += 1
                delta_quat = obs.gripper_pose[3: 7]
            gripper_pos = [1]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
        elif self._graspState == 2:
            # Grasping object
            delta_pos = obs.gripper_pose[: 3]
            delta_quat = obs.gripper_pose[3: 7]
            gripper_pos = [1]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
            res[2] = 0.75
            diff = obs.gripper_pose-res[: 7]
            if np.allclose(diff, np.zeros(7), atol=0.05):
                self._graspState += 1
                self._countPull = 0
                res[-1] = 0
                print("Grasped object!")
            else:
                print("Grasping object!")
        elif self._graspState == 3:
            depth = np.array(obs.wrist_depth)
            (height, width) = depth.shape
            if self._countPull < 40 and np.min(depth[height*1//2: height-25, 20: width-20]) < 0.01:
                # Pull back
                self._countPull += 1
                delta_pos = obs.gripper_pose[: 3]
                delta_quat = obs.gripper_pose[3: 7]
                gripper_pos = [0]
                res = np.concatenate(
                    [delta_pos, delta_quat, gripper_pos], axis=-1)
                res[2] = 0.9
                diff = obs.gripper_pose-res[: 7]
                if np.allclose(diff, np.zeros(7), atol=0.05):
                    self._graspState += 1
                    print("Pulled back, go to target now")
                else:
                    print("Pulling back")
            else:
                self._graspState = 1
                self.isRotated = False
                self.isTranslated = False
                delta_pos = obs.gripper_pose[: 3]
                delta_quat = obs.gripper_pose[3: 7]
                gripper_pos = [1]
                res = np.concatenate(
                    [delta_pos, delta_quat, gripper_pos], axis=-1)
                res[2] = 0.9
                print("Not picked, Trying again")
        elif self._graspState == 4:
            # Move grasped object
            delta_pos = self._targetPosition
            delta_quat = obs.gripper_pose[3: 7]
            gripper_pos = [0]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
            res[2] = 0.9
            diff = obs.gripper_pose-res[: 7]
            if np.allclose(diff, np.zeros(7), atol=0.05):
                self.reset()
                res[-1] = 1
                print("Moved to target, drop the object")
            else:
                print("Moving to target")
        elif self._graspState == 10:
            delta_pos = obs.gripper_pose[: 3]
            delta_quat = obs.gripper_pose[3: 7]
            gripper_pos = [0]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
            res[2] = 0.75
            diff = obs.gripper_pose-res[: 7]

            if np.allclose(diff, np.zeros(7), atol=0.05):
                print("Grasped object 10!")
                self._graspState += 1

        elif self._graspState == 11:
            print(self._sourcePosition)
            delta_pos = self._sourcePosition - obs.gripper_pose[:3]
            new_pose = obs.gripper_pose[:3] + delta_pos*0.1
            if np.allclose(new_pose, obs.gripper_pose[:3]):
                print("Done 11")
                self._graspState = 0
            res = np.concatenate([new_pose, obs.gripper_pose[3:7], [0]], axis=-1)



        return res


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0]*3# [0.005] * 3
        self._rot_scale = [0]*3#[0.01] * 3

    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(
            exclude_base=True, first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(
                self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            pose[: 3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y,
                        perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses


def simulation():
    # See rlbench/action_modes.py for other action modes
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
    action_mode_v = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(action_mode_v, '', ObservationConfig(), False)
    task = env.get_task(EmptyContainer)
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    descriptions, obs = task.reset()
    allPoses = obj_pose_sensor.get_poses()
    all_object_poses = [allPoses["Shape0"],allPoses["Shape1"], allPoses["Shape4"]]
    all_object_poses.sort(key=lambda x: np.linalg.norm(x[:3] - allPoses["large_container"][:3]))
    agent = RandomAgent(allPoses["large_container"],
                        allPoses["small_container0"], env)
    print(descriptions)
    obj_poses = obj_pose_sensor.get_poses()
    training_steps = 10000
    episode_length = 1
    i = 0
    while i < training_steps:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()
        mask = env._scene._cam_wrist_mask.capture_rgb()
        depth = (obs.wrist_depth, obs.wrist_rgb, mask)
        if agent.isResetting:
            #if agent._graspState == 1:
                #q.put(depth)
            nextState = agent.resetAct(obs, obj_poses)
            obs, reward, terminate = task.step(nextState)
            if not agent.isResetting:
                task._action_mode = action_mode_v
                env._action_mode = action_mode_v
                env._set_arm_control_action()
        else:
            if i > 0 and i % episode_length == 0:
                print(i, 'Reset Episode')
                agent.isResetting = True
                task._action_mode = action_mode
                env._action_mode = action_mode
                env._set_arm_control_action()
                agent.updateV()
                continue
            oldState, action = agent.act(obs)
            obs, reward, terminate = task.step(action)
            print(reward)
            nextState = np.zeros(8)
            nextState[:7] = obs.joint_positions
            nextState[7] = obs.gripper_open > 0.5
            agent.updateQ(oldState, action, nextState, reward)
            i += 1
    env.shutdown()


def processImage(q):
    plt.figure()
    while True:
        depth, rgb, mask = q.get()
        mask = mask[:, : ,0]*255
        plt.clf()
        # plt.imshow(mask)
        # plt.draw()
        # plt.pause(0.001)

        data = locateObject(depth)
        if data != None:
            depth, thres, center, delta, angle = data
            ret = DetectEdge(center, mask)
            if ret != None:
                _, _, point = ret
                plt.plot(point[0], point[1], 'bo--', linewidth=2, markersize=12)
            plt.plot(center[0], center[1], 'ro--', linewidth=2, markersize=12)
            plt.draw()
            plt.pause(0.001)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            blurred = (blurred*255).astype(np.uint8)
            canny = cv2.Canny(blurred,0,255)
            canny_show = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            cv2.imshow('image', np.hstack([rgb, canny_show]))
            cv2.waitKey(1)

def DetectEdge(center, mask):
    # cv2.imshow('mask', mask)
    returnval = None
    pack, mask = bfs_lookup(center, mask, 0)
    if (pack != None):
        i_o, j_o = pack
        #plt.plot(pack[0], pack[1], 'go--', linewidth=2, markersize=12)
        pack, mask = bfs_lookup((i_o, j_o), mask, 0)
        ref_val = mask[j_o, i_o]
        mask[mask==0]=ref_val
        if (pack != None):
            i, j = pack
            dis = (i-center[0])**2 + (j-center[1])**2
            dir = np.arctan2(center[0]-i, center[1]-j)*180/np.pi
            print("=======================================")
            print("distance:", dis, " direction:", dir, " i:", i, " j:", j)
            returnval = dis, dir, (i,j)
        else: 
            print("container bin edge not found ")
    else:
        print("located object edge not found!")

    return returnval


def bfs_lookup(center, mask, val):
    h, w = mask.shape
    ref = mask[center[1] , center[0]]
    q = [(center[0], center[1])]
    returnval = None 
    while q:
        i, j = q.pop(0)
        if 0 <= j < h and 0 <= i < w and mask[j][i] != val:
            if mask[j][i] == ref:
                mask[j][i] = val 
                q.append((i+1, j))
                q.append((i-1, j))
                q.append((i, j+1))
                q.append((i, j-1))
            elif mask[j][i] != val and returnval == None:
                returnval = (i , j)
    return returnval, mask
             
if __name__ == "__main__":
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=simulation, args=(queue,))
    p.start()
    p2 = multiprocessing.Process(target=processImage, args=(queue,))
    p2.start()