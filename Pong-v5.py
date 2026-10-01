#!/usr/bin/env python
# coding: utf-8
import gymnasium as gym
import ale_py
import torch
import torchvision.transforms as T
from PIL import Image
from collections import deque
import collections
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from tqdm import tqdm
matplotlib.use('Agg')
device = torch.device("cpu")

print(device)

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

    def __len__(self):
        return len(self.memory)

def select_action(state, epsilon, model, n_actions):
    model.eval()
    sample = random.random()

    if sample > epsilon:
        device = next(model.parameters()).device
        state_tensor = state.to(device).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax(dim=1).item()
    else:
        action = random.randrange(n_actions)

    return action

def optimize_model(policy_net, target_net, replay_buffer, optimizer, loss_fn, batch_size, gamma, device):
    if len(replay_buffer) < REPLAY_START_SIZE: 
        return
    
    batch = replay_buffer.sample(batch_size)
    unzipped_batch = zip(*batch)
    list_of_states, list_of_actions, list_of_rewards, list_of_next_states, list_of_done = unzipped_batch

    list_of_states = torch.stack(list_of_states).to(device)
    list_of_next_states = torch.stack(list_of_next_states).to(device)

    list_of_states = list_of_states.float() / 255.0
    list_of_next_states = list_of_next_states.float() / 255.0

    list_of_actions = torch.tensor(list_of_actions, dtype=torch.int64, device=device).unsqueeze(1)
    list_of_rewards = torch.tensor(list_of_rewards, dtype=torch.float32, device=device).unsqueeze(1)
    list_of_done = torch.tensor(list_of_done, dtype=torch.float32, device=device).unsqueeze(1)

    all_current_q_values = policy_net(list_of_states)
    current_q_values = all_current_q_values.gather(1, list_of_actions)

    with torch.no_grad():
        max_next_q = target_net(list_of_next_states).max(dim=1).values.unsqueeze(1)

        target_q  = list_of_rewards + (gamma * max_next_q * (1 - list_of_done))
    
    policy_net.train()
    loss = loss_fn(current_q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)

    optimizer.step()

import gymnasium as gym
preprocess_to_pil = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84, 84), antialias=True) 
])
def preprocess_frame(frame, device="cpu"):
    pil_img = preprocess_to_pil(frame)
    
    np_img = np.array(pil_img, dtype=np.uint8)
    
    np_img = np.expand_dims(np_img, axis=0)
    
    tensor_img = torch.from_numpy(np_img)
    
    return tensor_img.to(device)

class PongQNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PongQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc_q_values = nn.Linear(in_features=512, out_features=action_dim)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        
        q_values = self.fc_q_values(x)

        return q_values

MAX_GRAD_NORM = 10.0

batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 1000000
target_update_interval = 10000
replay_capacity = 50000 
learning_rate = 0.0001
max_steps_per_episode = 10000
REPLAY_START_SIZE = 10000 

action_map = {
    0: 0,
    1: 2,
    2: 3
}

n_actions = len(action_map)

pong_env = gym.make("ALE/Pong-v5", frameskip=4)
replay_buffer = ReplayBuffer(replay_capacity)

policy_net = PongQNetwork(n_actions).to(device)
target_net = PongQNetwork(n_actions).to(device)

optimizer = optim.Adam(policy_net.parameters(), lr = learning_rate)
loss_fn = nn.SmoothL1Loss()

total_steps = 0

all_episode_rewards = []
PLOT_EVERY_N_EPISODES = 50
all_episode_rewards = []
mean_rewards_history = []
steps_history = []
best_mean_reward = -float('inf')

checkpoint_path = "checkpoint_pong.pth"

start_episode = 0

if os.path.exists(checkpoint_path):
    print("Loading Pong checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only= False)
    policy_net.load_state_dict(checkpoint['policy_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    total_steps = checkpoint['total_steps']
    start_episode = checkpoint['episode'] + 1
    all_episode_rewards = checkpoint['all_episode_rewards']
    mean_rewards_history = checkpoint.get('mean_rewards_history', [])
    steps_history = checkpoint.get('steps_history', [])
    best_mean_reward = checkpoint.get('best_mean_reward', -float('inf'))
    
    if 'replay_buffer' in checkpoint:
        replay_buffer.memory = checkpoint['replay_buffer']
    
    print(f"Resuming from episode {start_episode} with {len(replay_buffer)} experiences loaded.")
else:
    total_steps = 0
    all_episode_rewards = []
    mean_rewards_history = []
    steps_history = []
    best_mean_reward = -float('inf')

num_episodes = 1900
mean_reward = -float('inf')
for episode in tqdm(range(start_episode, num_episodes), unit="epochs"):
    observation, info = pong_env.reset()

    frame_stack = deque(maxlen=4)
    processed_frame = preprocess_frame(observation, "cpu")

    for _ in range(4):
        frame_stack.append(processed_frame)

    state = torch.cat(list(frame_stack), dim = 0)

    episode_reward = 0
    for t in range(max_steps_per_episode):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1. * total_steps / epsilon_decay)
        action_index = select_action(state, epsilon, policy_net, n_actions)

        real_action = action_map[action_index]

        next_observation, reward, done, truncated, info = pong_env.step(real_action)
        episode_reward += reward

        new_processed_frame = preprocess_frame(next_observation, "cpu")
        frame_stack.append(new_processed_frame)
        next_state = torch.cat(list(frame_stack), dim = 0)

        replay_buffer.push(state, action_index, reward, next_state, done)

        state = next_state

        if len(replay_buffer) > REPLAY_START_SIZE:
            optimize_model(policy_net, target_net, replay_buffer, optimizer, loss_fn, batch_size, gamma, device)

        total_steps+=1

        if total_steps % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done or truncated:
            break
    all_episode_rewards.append(episode_reward)
    print(f"reward at episode {episode}/{num_episodes} is {episode_reward}", flush=True)
    if (episode + 1) % PLOT_EVERY_N_EPISODES == 0:
        mean_reward = np.mean(all_episode_rewards[-PLOT_EVERY_N_EPISODES:])
        mean_rewards_history.append(mean_reward)
        steps_history.append(total_steps) 
        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            
        print(f"Episode {episode + 1} | Total Steps: {total_steps} | Mean Reward (last {PLOT_EVERY_N_EPISODES}): {mean_reward:.2f}")
    if (episode+1) % 25 == 0:
        torch.save({
            'episode': episode,
            'total_steps': total_steps,
            'policy_net': policy_net.state_dict(),
            'target_net': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'all_episode_rewards': all_episode_rewards,
            'mean_rewards_history': mean_rewards_history,
            'steps_history': steps_history,
            'best_mean_reward': best_mean_reward,
            'replay_buffer': replay_buffer.memory 
        }, checkpoint_path)

print("Training complete")

plt.figure(figsize=(12, 6))
plt.plot(steps_history, mean_rewards_history, label=f'Mean {PLOT_EVERY_N_EPISODES}-Episode Reward')
plt.axhline(y=best_mean_reward, color='r', linestyle='--', label=f'Best Mean Reward: {best_mean_reward:.2f}')
plt.title('Final DQN Learning Curve - Pong')
plt.xlabel('Total Timesteps')
plt.ylabel(f'Mean {PLOT_EVERY_N_EPISODES}-Episode Reward')
plt.legend()
plt.grid(True)
plt.savefig('dqn_pong_learning_curve.png') 
plt.show()