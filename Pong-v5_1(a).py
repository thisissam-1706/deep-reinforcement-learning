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

pong_env = gym.make("ALE/Pong-v5", render_mode = None)

print(f"Observation space: {pong_env.observation_space}")
print(f"-----------------------------------------------------")
print(f"Action space: {pong_env.action_space}")
pong_env = gym.make("ALE/Pong-v5")

device = torch.device("cpu")
print("Using:", device)

observation, info = pong_env.reset()
timestep = 0

while True:
    obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1)

    obs_tensor /= 255.0

    action = pong_env.action_space.sample()
    action_tensor = torch.tensor(action, dtype=torch.int64, device=device)

    observation, reward, terminated, truncated, info = pong_env.step(action_tensor.item())

    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

    print(f"Reward at time step {timestep}: {reward_tensor.item()}")
    timestep += 1

    if terminated or truncated:
        break

pong_env.close()
print(observation.shape)
