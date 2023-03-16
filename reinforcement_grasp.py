'''
 - This script performs the grasping of all objects in the environment and
   drops them in the target bin and back to the environment.
 - It uses image feedback from on robot gripper and generates a reward for 
   reiforcement learning to locate and grasp objects.
'''

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
from quaternion import from_rotation_matrix, quaternion, as_float_array, from_euler_angles, rotation_chordal_distance
import cv2
import multiprocessing
from pyquaternion import Quaternion
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from rlbench.backend import task
import gym
from gym import spaces
import time


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


def locateObject(rgb):
    thresh = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)[:, :, 0]
    (height, width) = thresh.shape
    thresh = ((thresh/np.max(thresh))*255).astype('uint8')
    center = [thresh.shape[1]//2, thresh.shape[0]//2]

    # proc = cv2.adaptiveThreshold(
    #     depth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # thresh = cv2.bitwise_not(proc, proc)
    diff = 20
    for h in range(height):
        cv2.floodFill(thresh, None, (h, 0), 0, loDiff=diff, upDiff=diff)
        cv2.floodFill(thresh, None, (h, width - 1),
                      0, loDiff=diff, upDiff=diff)

    for w in range(width):
        cv2.floodFill(thresh, None, (0, w), 0, loDiff=diff, upDiff=diff)
        cv2.floodFill(thresh, None, (height - 1, w),
                      0, loDiff=diff, upDiff=diff)

    # kernel = np.ones((2, 2), np.uint8)
    # thresh = cv2.dilate(
    #     cv2.erode(thresh, kernel, iterations=1), kernel, iterations=4)

    # for h in range(height):
    #     cv2.floodFill(thresh, None, (h, 0), 0)
    #     cv2.floodFill(thresh, None, (h, width - 1), 0)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    numCount = len(contours)
    minIdx = -1
    minDist = None
    minCenter = None
    minPosIdx = None
    allPossible = []
    for i in range(numCount):
        c = contours[i]
        (x, y), (cw, ch), _ = cv2.minAreaRect(c)
        area = cv2.contourArea(c)
        if area > width*height/3 or max(cw, ch) > max(width, height) * 3 / 4 or max(cw, ch) < max(width, height) / 10:
            continue
        cX = int(np.average(c[:, :, 0]))
        cY = int(np.average(c[:, :, 1]))
        distance = sp.spatial.distance.euclidean(center, [cX, cY])
        allPossible.append((cX, cY))
        if minIdx == -1 or minDist > distance:
            minDist = distance
            minIdx = i
            minCenter = (cX, cY)
            minPosIdx = len(allPossible) - 1
    if minDist == None:
        return rgb, thresh, None, None, None, None

    rect = cv2.minAreaRect(contours[minIdx])
    (x, y), (width, height), angle = rect

    allDistances = []
    for i in range(len(allPossible)):
        if i != minPosIdx:
            (cX, cY) = allPossible[i]
            allDistances.append(np.linalg.norm([cX - x, cY - y]))

    minClosest = None
    if len(allDistances) > 0:
        minClosest = np.amin(allDistances)

    if (width < height):
        angle -= 90
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(rgb, contours, -1, (255, 0, 0), 1)
    cv2.drawContours(rgb, [box], 0, (0, 255, 0), 2)
    return rgb, thresh, minCenter, np.array(minCenter) - np.array(center), angle + 90, minClosest


def discrete(state):
    t = np.zeros(8)
    t[:7] = np.around(np.rad2deg(state[:7]))
    t[7] = state[7] > 0.5
    return np.int_(t)


class RandomAgent:

    def __init__(self, sourceBin, targetBin, env):
        self._env = env
        self._sourcePosition = np.copy(sourceBin)
        self._sourcePosition[2] = 1
        self._targetPosition = np.copy(targetBin)
        self._targetPosition[2] = 0.9
        self.reset()
        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3
        self.policy = {}
        self.q = {}
        self.value = {}
        self.alpha = 0.5
        self.epsilon = 0.2
        self.gamma = 0.95
        self.angleOffset = 0
        self.isResetting = False
        self.action_space = spaces.Box(
            low=np.concatenate([np.ones(7)*(-1), [0]], axis=-1), high=np.concatenate([np.ones(7), [1]], axis=-1), dtype=np.int)

    def reset(self):
        self._graspState = 0
        self.angleOffset = 0
        self.isRotated = False
        self.isTranslated = False
        self.isFinalized = False
        self.lastPose = np.zeros(11)

    def act(self, obs):
        current = np.concatenate(
            [obs.joint_positions, [obs.gripper_open]], axis=-1)
        action = self.action_space.sample()
        if np.random.random_sample() > self.epsilon and hash(str(discrete(current))) in self.policy:
            temp = self.policy.get(hash(str(discrete(current))), np.zeros(8))
            if np.linalg.norm(temp) > 0:
                print(discrete(current))
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

    def moveDownWithGripper(self, obs, gripperOpen):
        delta_pos = obs.gripper_pose[: 3]
        delta_quat = obs.gripper_pose[3: 7]
        gripper_pos = [gripperOpen]
        res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
        res[2] = 0.75
        return res

    def resetAct(self, obs, poses):
        print("state", self._graspState)
        if self._graspState == 0:
            # Move to bin
            delta_pos = self._sourcePosition[:3]
            delta_quat = R.from_euler('y', 180, degrees=True).as_quat()
            gripper_pos = [obs.gripper_pose[-1]]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
            diff = obs.gripper_pose-res[: 7]
            if np.allclose(diff, np.zeros(7), atol=0.05):
                self._graspState += 1
                res[-1] = 1
                self.isRotated = False
                self.isTranslated = False
                self.isFinalized = False
                print("Move close to grasp object now")
        elif self._graspState == 1:
            # Locate object
            depth, thres, center, delta, angle, closestDistance = locateObject(
                obs.wrist_rgb)
            if center == None:
                self.reset()
                print("Switch Bin")
                temp = self._sourcePosition[:3]
                self._sourcePosition = self._targetPosition
                self._targetPosition = temp
                res = np.concatenate([obs.gripper_pose, [1]], axis=-1)
                return res
            delta = np.array(delta)/1000
            delta_pos = obs.gripper_pose[: 3]
            if not self.isTranslated:
                print("Translating Gripper")
                mask = (np.abs(delta) > 0.01)
                delta = delta * mask
                if np.allclose(delta, np.zeros(2), atol=0.001):
                    self.isTranslated = True
                else:
                    rotationM = np.eye(4)
                    rotationM[: 3, : 3] = R.from_quat(
                        obs.gripper_pose[3: 7]).as_matrix()
                    rotationM[: 3, 3] = np.array(obs.gripper_pose[: 3])
                    p = np.array([delta[1], -delta[0], 0, 1])
                    p = rotationM@p
                    delta_pos = p[: 3]
                delta_quat = R.from_euler('y', 180, degrees=True).as_quat()
            elif not self.isRotated:
                targetR = R.from_euler(
                    'z', angle + self.angleOffset, degrees=True)
                nextRotationR = R.from_quat(
                    obs.gripper_pose[3: 7]).as_matrix()@targetR.as_matrix()
                delta_quat = obs.gripper_pose[3: 7]
                delta_quat = R.from_matrix(nextRotationR).as_quat()
                self.isRotated = True
                print("Rotating Gripper")
            elif not self.isFinalized:
                print("Adjusting Gripper Location")
                mask = (np.abs(delta) > 0.005)
                delta = delta * mask
                if np.allclose(delta, np.zeros(2), atol=0.0005):
                    mask1 = self._env._scene._cam_wrist_mask.capture_rgb()
                    mask1 = mask1[:, :, 0]*255
                    ret = DetectEdge(center, mask1)
                    dis = None
                    if ret != None:
                        _, dis, direction, point = ret
                    if dis == None or dis > 75:
                        self.isFinalized = True
                    else:
                        self._graspState = 1.5
                        delta_quat = obs.gripper_pose[3: 7]
                else:
                    rotationM = np.eye(4)
                    rotationM[: 3, : 3] = R.from_quat(
                        obs.gripper_pose[3: 7]).as_matrix()
                    rotationM[: 3, 3] = np.array(obs.gripper_pose[: 3])
                    p = np.array([delta[1], -delta[0], 0, 1])
                    p = rotationM@p
                    delta_pos = p[: 3]
                delta_quat = obs.gripper_pose[3: 7]
            else:
                self._graspState += 1
                delta_quat = obs.gripper_pose[3: 7]
            gripper_pos = [1]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
        elif self._graspState == 1.5:
            # Declutter the object
            delta_pos = obs.gripper_pose[: 3]
            delta_quat = obs.gripper_pose[3: 7]
            gripper_pos = [0]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
            res[2] = 0.75
            diff = obs.gripper_pose-res[: 7]
            if np.allclose(diff, np.zeros(7), atol=0.05):
                print("Grasped object 10!")
                self._graspState = 1.75

        elif self._graspState == 1.75:
            delta_pos = np.copy(self._sourcePosition[:3])
            delta_pos[2] = obs.gripper_pose[2]
            if np.allclose(delta_pos, obs.gripper_pose[:3], atol=0.05):
                print("Done 11")
                self._graspState = 0
            res = np.concatenate(
                [delta_pos, obs.gripper_pose[3:7], [0]], axis=-1)
        elif self._graspState == 2:
            # Grasping object
            print(np.linalg.norm(
                obs.gripper_pose[:3] - self._sourcePosition[:3]))
            res = self.moveDownWithGripper(obs, gripperOpen=True)
            diff = obs.gripper_pose-res[: 7]
            if np.allclose(diff, np.zeros(7), atol=0.05):
                self._graspState += 1
                self.graspedTime = time.perf_counter()
                res[-1] = 0
                print("Grasped object!")
            else:
                print("Grasping object!")
        elif self._graspState == 3:
            depth = np.array(obs.wrist_depth)
            (height, width) = depth.shape
            if time.perf_counter() - self.graspedTime < 30 and np.min(depth[height*1//2: height-25, 20: width-20]) < 0.01:
                # Pull back
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
                print("Depth check", np.min(
                    depth[height*1//2: height-25, 20: width-20]))
                self._graspState = 1
                self.angleOffset += 90
                self.isRotated = False
                self.isTranslated = False
                self.isFinalized = False
                delta_pos = obs.gripper_pose[: 3]
                delta_quat = obs.gripper_pose[3: 7]
                gripper_pos = [1]
                self._env._robot.gripper.release()
                res = np.concatenate(
                    [delta_pos, delta_quat, gripper_pos], axis=-1)
                res[2] = 0.9
                print("Not picked, Trying again")
        elif self._graspState == 4:
            # Move grasped object
            delta_pos = self._targetPosition[:3]
            delta_quat = R.from_euler('y', 180, degrees=True).as_quat()
            gripper_pos = [0]
            res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
            diff = obs.gripper_pose-res[: 7]
            if np.allclose(diff, np.zeros(7), atol=0.05):
                self.reset()
                res[-1] = 1
                self._env._robot.gripper.release()
                print("Moved to target, drop the object")
            else:
                print("Moving to target")

        return res


class Reward:

    def __init__(self, env, objectNames, targetBinName):
        self._env = env
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(
            exclude_base=True, first_generation_only=False)
        for obj in objs:
            name = obj.get_name()
            if name == targetBinName:
                self._targetLocation = obj.get_pose()[:3]
        self._objectNames = objectNames
        self.objectsToBinWeight = 5
        self.gripperToObjectWeight = 3
        self.gripperObjectInGripper = 4
        self.orientationWeight = 0.1
        self.gripperClosedWeight = 1
        self.desiredR = R.from_euler(
            'y', 180, degrees=True).as_euler('ZXZ', degrees=True)

    def evaluate(self, obs):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(
            exclude_base=True, first_generation_only=False)
        reward = 0
        if obs.gripper_open == 0:
            reward -= self.gripperClosedWeight
        for obj in objs:
            name = obj.get_name()
            if name in self._objectNames:
                objectPos = obj.get_pose()[:3]
                objToBin = np.linalg.norm(
                    objectPos[:2]-self._targetLocation[:2])
                reward += self.objectsToBinWeight * objToBin
                dist = np.linalg.norm(
                    objectPos - obs.gripper_pose[: 3])*objToBin
                reward += self.gripperToObjectWeight*dist
                if np.allclose(dist, np.zeros(7), atol=0.05):
                    reward -= self.gripperObjectInGripper * \
                        (1 - obs.gripper_open)

        if reward > 0:
            reward = 1/reward
        return reward


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

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


def simulation(q):
    # See rlbench/action_modes.py for other action modes
    withLearning = False
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN)
    action_mode_v = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    if withLearning:
        env = Environment(action_mode_v, '', ObservationConfig(), False)
    else:
        env = Environment(action_mode, '', ObservationConfig(), False)
    task = env.get_task(EmptyContainer)
    obj_pose_sensor = NoisyObjectPoseSensor(env)
    descriptions, obs = task.reset()
    allPoses = obj_pose_sensor.get_poses()
    agent = RandomAgent(allPoses["large_container"],
                        allPoses["small_container0"], env)
    print(descriptions)
    obj_poses = obj_pose_sensor.get_poses()
    training_steps = 10000
    episode_length = 100
    i = 0
    rewardFunc = Reward(
        env, ['Shape0', 'Shape2', 'Shape4'], "small_container0")
    while i < training_steps:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()
        mask = env._scene._cam_wrist_mask.capture_rgb()
        data = (obs.wrist_depth, obs.wrist_rgb, mask)
        if withLearning:
            if i > 0 and i % episode_length == 0:
                print(i, 'Reset Episode')
                agent.updateV()
                i += 1
                continue
            oldState, action = agent.act(obs)
            obs, _, terminate = task.step(action)
            reward = rewardFunc.evaluate(obs)
            print(reward)
            nextState = np.zeros(8)
            nextState[:7] = obs.joint_positions
            nextState[7] = obs.gripper_open > 0.5
            agent.updateQ(oldState, action, nextState, reward)
            i += 1
        else:
            q.put(data)
            nextState = agent.resetAct(obs, obj_poses)
            obs, reward, terminate = task.step(nextState)
        # if terminate:
        #     break
    env.shutdown()


def viewImage(image):
    cv2.imshow('Display', image)
    cv2.waitKey(1)


def DetectEdge(center, mask, show=False):
    returnval = None
    pack, mask = bfs_lookup(center, mask, 0)
    if (pack != None):
        i_o, j_o = pack
        #plt.plot(pack[0], pack[1], 'go--', linewidth=2, markersize=12)
        pack, mask = bfs_lookup((i_o, j_o), mask, 0)
        ref_val = mask[j_o, i_o]
        mask[mask == 0] = ref_val
        if (pack != None):
            i, j = pack
            dis = (i-center[0])**2 + (j-center[1])**2
            dir = np.arctan2(center[0]-i, center[1]-j)*180/np.pi
            print("=======================================")
            print("distance:", dis, " direction:", dir, " i:", i, " j:", j)
            returnval = mask, dis, dir, (i, j)
        else:
            print("container bin edge not found ")
    else:
        print("located object edge not found!")

    return returnval


def processImage(q):
    while True:
        depth, rgb, mask = q.get()
        mask = mask[:, :, 0]*255
        rgb, thres, center, delta, angle, _ = locateObject(rgb)
        if center != None:
            ret = DetectEdge(center, mask, show=True)
            if ret != None:
                mask, _, _, _ = ret
        viewImage(
            np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(thres, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))


def bfs_lookup(center, mask, val):
    h, w = mask.shape
    ref = mask[center[1], center[0]]
    q = [(center[0], center[1])]
    print(q)
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
                returnval = (i, j)
    # plt.imshow(mask)
    return returnval, mask


if __name__ == "__main__":

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=simulation, args=(queue,))
    p.start()
    p2 = multiprocessing.Process(target=processImage, args=(queue,))
    p2.start()
