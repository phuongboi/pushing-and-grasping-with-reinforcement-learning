### This work is based on the project [visual-pushing-grasping](https://github.com/andyzeng/visual-pushing-grasping) control UR5 robot in CoppeliaSim(V-REP)
* I do some major changes focus on reducing computation complexity by using lightweight network and a different way of modeling action space, reward.
#### [10/12/2023] Update test script and weight
* Update test script and pretrained weight
* Test result [video](https://github.com/phuongboi/pushing-and-grasping-with-reinforcement-learning/blob/main/figures/recording_2023_12_10-15_55-17.avi)
#### [25/11/2023] Update two head grasp model
* Only for grasping action
* Using mobilenetv2 as backbone and 2 prediction head (1 for 16 orientation, 1 for 112x112 location of grasping action)
#### TODO:
* Update evalution script
* Using ROS replace V-REP python api
* Increase location map to 224x224 to improve precision
* Add more 1 prediction head for pushing/grasping

#### [24/11/2023] Single branch, end-to-end pipeline
* End-to-end pipeline, single branch, replace densenet121 with mobilenetv2
* No input rotation, modeling action space as a 3D tensor 112x112x16 (height map resolution=4mm, 8 angle rotations)
* Only use RGB as input of network, depth information for z position
##### CoppeliaSim simulation
* The simulation scene when training `train_twoheadgraspnet.py`, robot successfuly learn to find the object and do grasping action. Due to the limit of resolution (4mm instead of 2mm in original work), location prediction is sometime inaccurate. There is no pushing action so robot find difficult to handle complex scenerios. The scene is recorded during training phase so there are random actions in sequence of actions.
![alt text](https://github.com/phuongboi/pushing-and-grasping-with-reinforcement-learning/blob/main/figures/recording_2023_11_28-07_03-58.gif)

##### Training result
* Training result of two head grasp model
![alt text](https://github.com/phuongboi/pushing-and-grasping-with-reinforcement-learning/blob/main/figures/fig_14000.png)

### Requirements
* CoppeliaSim v4.5.1 linux
* Pytorch

### Setup
* Open simulation/simulation.ttt in CoppeliaSim
* Run `python train_twoheadgraspnet.py`
### Note
* This repository is under experimenting and developing period
* Need to do more expreriment with one head model
### Reference
* [1] https://github.com/andyzeng/visual-pushing-grasping
* [2] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
