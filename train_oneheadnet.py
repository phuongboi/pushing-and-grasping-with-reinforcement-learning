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
from models import OneHeadNetwork

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
        self.action_space = [112, 112, 16]
        self.input_shape = [224, 224]
        self.model = OneHeadNetwork()
    #
    # def make_model(self):
    #     model = Network()
    #     return model

    def agent_policy(self, state, epsilon):
        input_image, input_depth = self.transform_data(state)
        #print(input_image.shape, input_depth.shape)
        # epsilon greedy policy
        if np.random.rand() < epsilon:
            #action_orientaton_idx = np.random.randint(self.num_action_orientations)
            #action_position_idx = np.random.randint(self.num_action_positions, size=2)
            action = np.random.randint(self.num_actions)

            #return [action_position_idx[0], action_position_idx[1], action_orientaton_idx]
            return action

        else:
            # q_value = self.model(torch.FloatTensor(np.float32(state)).unsqueeze(0).cuda())
            # action = np.argmax(q_value.cpu().detach().numpy())
            #print("using model")
            q_value = self.model(torch.from_numpy(input_image), torch.from_numpy(input_depth))
            q_value = q_value.permute(0, 2, 3, 1)
            #print("q value shape", q_value.shape)
            # max_heatmap = np.max(q_value, axis=(1, 2))

            q_value = q_value.reshape(q_value.shape[0], q_value.shape[1]*q_value.shape[2]*q_value.shape[3])
            action = np.argmax(q_value.detach().numpy(), axis=1).squeeze()

        return action


    def add_to_replay_buffer(self, state, action, reward, next_state, terminal):
        input_image, input_depth = self.transform_data(state)
        next_input_image, next_input_depth = self.transform_data(next_state)
        input_depth = next_input_depth = np.ones((1,1))
        self.replay_buffer.append((input_image, input_depth, action, reward, next_input_image, next_input_depth, terminal))

    def sample_from_reply_buffer(self):
        random_sample = random.sample(self.replay_buffer, self.batch_size)
        return random_sample

    def get_memory(self, random_sample):
        input_images = np.array([i[0] for i in random_sample])
        input_depths = np.array([i[1] for i in random_sample])
        actions = np.array([i[2] for i in random_sample])
        rewards = np.array([i[3] for i in random_sample])
        next_input_images = np.array([i[4] for i in random_sample])
        next_input_depths = np.array([i[5] for i in random_sample])
        terminals = np.array([i[6] for i in random_sample])
        #return torch.FloatTensor(np.float32(states)).cuda(), torch.from_numpy(actions).cuda(), rewards, torch.FloatTensor(np.float32(next_states)).cuda(), terminals
        return torch.from_numpy(input_images).squeeze(1), torch.from_numpy(input_depths).squeeze(1), torch.from_numpy(actions), rewards, torch.from_numpy(next_input_images).squeeze(1), torch.from_numpy(next_input_depths).squeeze(1), terminals

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

    def train_with_relay_buffer(self):
            # replay_memory_buffer size check
        if len(self.replay_buffer) < self.batch_size:
            return

        # Early Stopping
        # if np.mean(self.rewards_list[-10:]) > 180:
        #     return
        sample = self.sample_from_reply_buffer()
        input_images, input_depths, actions, rewards, next_input_images, next_input_depths, terminals = self.get_memory(sample)
        next_q_mat = self.model(next_input_images, next_input_depths)
        next_q_mat = next_q_mat.permute(0, 2, 3, 1)
        next_q_mat = next_q_mat.reshape(next_q_mat.shape[0], next_q_mat.shape[1]*next_q_mat.shape[2]*next_q_mat.shape[3])
        #print("next_q_mat",next_q_mat.shape)


        # next_q_vec = np.max(next_q_mat.cpu().detach().numpy(), axis=1).squeeze()
        #
        # target_vec = rewards + self.gamma * next_q_vec* (1 - terminals)
        # q_mat = self.model(states)
        # q_vec = q_mat.gather(dim=1, index=actions.unsqueeze(1)).type(torch.FloatTensor).cuda()
        # target_vec = torch.from_numpy(target_vec).unsqueeze(1).type(torch.FloatTensor).cuda()
        next_q_vec = np.max(next_q_mat.detach().numpy(), axis=1).squeeze()
        #print("next_q_vec" ,next_q_vec.shape)
        target_vec = rewards + self.gamma * next_q_vec* (1 - terminals)
        q_mat = self.model(input_images, input_depths)
        q_mat = q_mat.permute(0, 2, 3, 1)
        q_mat = q_mat.reshape(q_mat.shape[0], q_mat.shape[1]*q_mat.shape[2]*q_mat.shape[3])
        q_vec = q_mat.gather(dim=1, index=actions.unsqueeze(1)).type(torch.FloatTensor)
        #print(q_vec.shape)
        target_vec = torch.from_numpy(target_vec).unsqueeze(1).type(torch.FloatTensor)
        loss = self.loss_func(q_vec, target_vec)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, num_episodes=2000):
        #self.model.cuda().train()
        self.model.train()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        steps_done = 0
        losses = []
        rewards_list = []
        for episode in range(num_episodes):
            state = env.reset()
            reward_for_episode = 0
            #num_steps = 1000
            #state = state[0]
            #for step in range(num_steps):
            num_step_per_eps = 0
            while True:
                epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(- steps_done / self.eps_decay)
                #print("CURRENT EPS", epsilon)
                #print("curent eps", epsilon)
                received_action = self.agent_policy(state, epsilon)
                steps_done += 1
                num_step_per_eps += 1
                # print("received_action:", received_action)
                next_state, reward, terminal = env.step(received_action, state[1], state[2])

                # Store the experience in replay memory
                self.add_to_replay_buffer(state, received_action, reward, next_state, terminal)
                # add up rewards
                reward_for_episode += reward
                state = next_state
                if len(self.replay_buffer) == self.initial_memory:
                    print("Start learning from buffer")
                if len(self.replay_buffer) > self.initial_memory and steps_done % 4 == 0:
                    loss = self.train_with_relay_buffer()
                    losses.append(loss.item())

                if steps_done % 400 == 0:
                    plot_stats(steps_done, rewards_list, losses, steps_done)
                    path = os.path.join(self.model_path, f"steps_{steps_done+1}.pth")
                    torch.save(self.model.state_dict(), path)

                if terminal:
                    rewards_list.append(reward_for_episode)
                    print("Episode: {} done, Reward: {}".format(episode, reward_for_episode))
                    break



            # Check for breaking condition
            # if (episode+1) % 800 == 0:
            #     path = os.path.join(self.model_path, f"{env.spec.id}_episode_{episode+1}.pth")
            #     print(f"Saving weights at Episode {episode+1} ...")
            #     torch.save(self.model.state_dict(), path)
        env.close()


def plot_stats(frame_idx, rewards, losses, step):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title(f'Total frames {frame_idx}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    #plt.show()
    plt.savefig('figures/fig_{}.png'.format(step))


if __name__ == "__main__":
    env = VrepEnvironment(is_testing=False,
                    )

    # setting up params
    lr = 0.0001
    batch_size = 8
    eps_decay = 30000
    eps_start = 1
    eps_end = 0.1
    initial_memory = 500
    memory_size = 3000#20 * initial_memory
    gamma = 0.99 # discount 0.5
    num_episodes = 500
    model_path = "weights/"
    print('Start training')
    model = DQN(model_path, env, lr, batch_size, gamma, eps_decay, eps_start, eps_end,initial_memory, memory_size)
    model.train(num_episodes)
