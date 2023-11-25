### (On-going) This repository is my re-implementation of the project [visual-pushing-grasping](https://github.com/andyzeng/visual-pushing-grasping), focusing in simulation in CoppeliaSim.

#### [25/11/2023] Update two head grasp model
* Only for grasping action
* Using resnet18 as backbone and 2 prediction head (1 for orientation, 1 for location of grasping action)

#### [24/11/2023] Init commit
* Using light weight model mobilenetv2 replace for densnet121
* No input rotation, reduce action space: 112x112x16 (height map resolution=4mm, 8 angle rotations)
* Only use RGB input
##### CoppeliaSim simulation
<!---![alt text](https://github.com/phuongboi/land-following-with-reinforcement-learning/blob/main/figures/recording_2023_10_19-06_46-39.gif) -->

##### Training result
<!---![alt text](https://github.com/phuongboi/land-following-with-reinforcement-learning/blob/main/figures/fig_40000.png)  -->

### Requirements
* CoppeliaSim v4.5.1 linux
* Pytorch

### Setup
* Open simulation/simulation.ttt in CoppeliaSim
* Run `python train_oneheadnet.py` or `python train_twoheadgraspnet.py`
### Note
* This repository is under experimenting and developing period
### Reference
* [1] https://github.com/andyzeng/visual-pushing-grasping
* [2] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
