### (On-going)This repository is my re-implementation of the project [visual-pushing-grasping](https://github.com/andyzeng/visual-pushing-grasping), focusing in simulation in CoppeliaSim.

#### [24/11/2023] Init commit
<!---
* Using CoppeliaSim(V-REP), ROS, Q-learning
* Simple and friendly implementation with pytorch
* Modify the ROS interface with new V-REP version

##### CoppeliaSim simulation
<!---![alt text](https://github.com/phuongboi/land-following-with-reinforcement-learning/blob/main/figures/recording_2023_10_19-06_46-39.gif)

##### Training result
![alt text](https://github.com/phuongboi/land-following-with-reinforcement-learning/blob/main/figures/fig_40000.png)  

### Requirements
* CoppeliaSim v4.5.1 linux
* Pytorch

### Setup
* Launch `roscore` in one terminal before launch Coppeliasim in another terminal to make sure that CoppeliaSim can load ROS plugin properly
* Open v_rep_scenario/scenario1.ttt in CoppeliaSim and modify child_script of Pioneer_p3dx by v_rep_scenario/rosInterfaceScript.lua
* Start CoppeliaSim simulation, make sure topics is work as expect by `rostopic list`
* Run `python train_qnetwork.py` -->

### Reference
* [1] https://github.com/andyzeng/visual-pushing-grasping
* [2] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
