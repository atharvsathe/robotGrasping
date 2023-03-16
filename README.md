# CMU - Robot Autonomy (16-662) Spting 2020 Simulation Project

## Problem - EmptyContainer
The goal of this task is bin picking, which is what many robotic startups are working on such ascovariant.ai, https://www.iamrobotics.com/, and https://www.berkshiregrey.com/to name a few. You will need to move objects to containers on the side. Then once the task has been completed, you will need to put the objects back into the original container. Some additional options include adding more random objects every so often.

## Our Approach
1. Using noisy poses of the objects from simulation. Uses gripper feedback for check for successful grasps. (gripper_feedback.py)
2. Using rgbd bird's eye view and robot gripper camera for setting and resetting the environment. (image_feedback.py)
Screen recording of the demo.
https://drive.google.com/file/d/1DjIh-1AnZRiam68kam5vjWtM9IQP3K9w/view?usp=sharing
3. Reinforcement learning approach. Inconclusive results. (reinforcement_grasp.py)

## Installation
Please use Python 3.6

1. Install [PyRep](https://github.com/stepjam/PyRep)
2. Install [RLBench](https://github.com/stepjam/RLBench)
3. `pip install -r requirements.txt`





