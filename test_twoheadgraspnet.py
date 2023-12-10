import numpy as np
from collections import deque
import random
import torch
from torch import nn
import os
from env import VrepEnvironment
#import params
from matplotlib import pyplot as plt
from IPython.display import clear_output
from scipy import ndimage
from models import OneHeadNetwork, TwoHeadGraspNetwork

class DQN:
    def __init__(self, model_path, env, lr, batch_size, gamma, eps_decay, eps_start, eps_end, initial_memory, memory_size):

        self.env = env
        self.model_path = model_path
        self.lr = lr
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.initial_memory = initial_memory

        self.replay_buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.num_actions = 112*112*16
        self.num_action_orientations = 16
        self.num_action_positions = 112
        self.input_shape = [224, 224]
        self.model = TwoHeadGraspNetwork()
        self.model.load_state_dict(torch.load(self.model_path))


    def agent_policy(self, state):
        input_image, input_depth = self.transform_data(state)

        q_value_orient, q_value_loc = self.model(torch.from_numpy(input_image), torch.from_numpy(input_depth))
        action_orientaton_idx = np.argmax(q_value_orient.detach().numpy()).squeeze()
        action_location_idx = np.argmax(q_value_loc.detach().numpy()).squeeze()

        action =  [action_location_idx, action_orientaton_idx]
        print(action)

        return action


    def transform_data(self, state):
        color_heightmap, valid_depth_heightmap, _ = state
        color_heightmap_2x = color_heightmap#ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = valid_depth_heightmap #ndimage.zoom(valid_depth_heightmap, zoom=[2,2], order=0)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        input_image = np.transpose(np.expand_dims(input_color_image.astype(np.float32), axis=0),(0, 3, 1, 2))
        input_depth = np.transpose(np.expand_dims(input_depth_image.astype(np.float32), axis=0), (0, 3, 1, 2))
        return input_image, input_depth


    def test(self):
        #self.model.cuda().train()
        self.model.eval()
        for episode in range(num_episodes):
            state = env.reset()
            reward_for_episode = 0
            while True:
                received_action = self.agent_policy(state)
                # print("received_action:", received_action)
                next_state, reward, terminal = env.twohead_grasp_step(received_action, state[1], state[2])
                # add up rewards
                reward_for_episode += reward
                state = next_state

                if terminal:
                    print("Episode: {} done, Reward: {}".format(episode, reward_for_episode))
                    break

        env.close()


if __name__ == "__main__":
    env = VrepEnvironment(is_testing=False,
                    )

    # setting up params
    lr = 0.0001
    batch_size = 8
    eps_decay = 30000
    eps_start = 0.5
    eps_end = 0.1
    initial_memory = 500
    memory_size = 1500#20 * initial_memory
    gamma = 0.5 # discount 0.5
    num_episodes = 600
    model_path = "weights/steps_10401.pth"
    print('Start evaluating')
    model = DQN(model_path, env, lr, batch_size, gamma, eps_decay, eps_start, eps_end,initial_memory, memory_size)
    model.test()
