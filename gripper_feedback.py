'''
 - This script performs the grasping of all objects in the environment and
   drops them in the target bin.
 - The logic uses noisy positions given by the simulation environment and 
   then plans the trajectory accordingly.
 - 4 states are defined for intermediate positions of the robotic arm.
 - The algorithm uses distance sensor of the gripper to check if the 
   grasp is successful.
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from quaternion import from_rotation_matrix, quaternion, as_float_array


from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *


def skew(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

# Samples a 6D pose from a zero-mean isotropic normal distribution
def sample_normal_pose(pos_scale, rot_scale):
    pos = np.random.normal(scale=pos_scale)
    eps = skew(np.random.normal(scale=rot_scale))
    R = sp.linalg.expm(eps)
    quat_wxyz = from_rotation_matrix(R)
    return pos, quat_wxyz

class RandomAgent:

    def __init__(self, allPoses, targetPose):
        self._allPositions = []
        for pose in allPoses:
            self._allPositions.append(pose[:3])
        self._targetPosition = targetPose[:3]
        self._current = 0 # Stores the object that is picked
        self._graspState = 0 # Stores the current state
        self.isDone = False # Checks if all tasks are done
            

    # Defines the move of the robot arm
    def make_move(obs, gripper_pos, res_2=0):
        if self._graspState < 3:
            delta_pos = self._allPositions[self._current]
        else :
            delta_pos = self._targetPosition

        delta_quat = obs.gripper_pose[3:7]
        res = np.concatenate([delta_pos, delta_quat, gripper_pos], axis=-1)
        res[2] += res_2
        diff = obs.gripper_pose - res[:7]
        return res, diff

    # Checks if the move has been completed
    def check_move(diff, state, message):
        if np.allclose(diff, np.zeros(7), atol=0.05):
                self._graspState = state
                print(message)

    # Main logic of the agent
    def act(self, obs, poses):
        if self._current >= len(self._allPositions):
            gripper_pos = [1]
            res = np.concatenate([obs.gripper_pose, gripper_pos], axis=-1)
            self.isDone = True
            return res

        # Move to object
        if self._graspState == 0:
            res, diff = make_move(obs, [0], 0.1)
            check_move(diff, 1, "Move close to grasp object now")
              
        # Grasping object  
        elif self._graspState == 1:
            res, diff = make_move(obs, [1], 0)
            check_move(diff, 2, "Grasped object!")
            res[-1] = 0

        # Pulling back while gripping
        elif self._graspState == 2:
            # Succefull grasp
            if  (obs.gripper_joint_positions[0] > 0.001 and obs.gripper_joint_positions[1] > 0.001):
                res, diff = make_move(obs, [0], 0.15)
                check_move(diff, 3, "Pulled back, go to target now")
 
            # Unseuccessful grasp
            else: 
                self._graspState = 0
                res, diff = make_move(obs, [0], 0)
                print("Not picked, Trying again")   

        # Move object to destination
        elif self._graspState == 3:
            res, diff = make_move(obs, [0], 0.15)
            check_move(diff, 0, "Moved to target, drop the object")
            res[-1] = 1
                    
        return res


class NoisyObjectPoseSensor:

    def __init__(self, env):
        self._env = env

        # Generate noise in the poses
        self._pos_scale = [0.005] * 3
        self._rot_scale = [0.01] * 3

    # Get poses of all objects and target positions
    def get_poses(self):
        objs = self._env._scene._active_task.get_base().get_objects_in_tree(exclude_base=True, first_generation_only=False)
        obj_poses = {}

        for obj in objs:
            name = obj.get_name()
            pose = obj.get_pose()

            pos, quat_wxyz = sample_normal_pose(self._pos_scale, self._rot_scale)
            gt_quat_wxyz = quaternion(pose[6], pose[3], pose[4], pose[5])
            perturbed_quat_wxyz = quat_wxyz * gt_quat_wxyz

            pose[:3] += pos
            pose[3:] = [perturbed_quat_wxyz.x, perturbed_quat_wxyz.y, perturbed_quat_wxyz.z, perturbed_quat_wxyz.w]

            obj_poses[name] = pose

        return obj_poses



if __name__ == "__main__":
    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN) # See rlbench/action_modes.py for other action modes
    env = Environment(action_mode, '', ObservationConfig(gripper_joint_positions=True,
        gripper_touch_forces=True), False)
    task = env.get_task(EmptyContainer)
    obj_pose_sensor = NoisyObjectPoseSensor(env)

    descriptions, obs = task.reset()

    allPoses = obj_pose_sensor.get_poses()
    allObjectPoses = [allPoses["Shape0"],allPoses["Shape1"], allPoses["Shape4"]]
    agent = RandomAgent(allObjectPoses, allPoses["small_container0"])

    while True:
        # Getting noisy object poses
        obj_poses = obj_pose_sensor.get_poses()

        # Perform action and step simulation
        action = agent.act(obs, obj_poses)
        obs, reward, terminate = task.step(action)
        if agent.isDone:
            break

    env.shutdown()
