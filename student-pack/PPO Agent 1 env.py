# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:29:58 2019

@author: Korstiaan
"""

#

import ipdb
import numpy as np

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
%matplotlib inline

import multiprocessing_env
from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
from multiprocessing_env import SubprocVecEnv
from torch.distributions import Categorical 


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

num_envs = 2

#Multiple environments giving shit - all subsequent code changed - where 'envs' - replaced with env 

#def make_env(i):
#    def _thunk():
#        env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=i, docker_training=False, retro=True,
#                       realtime_mode=False,
#                       config=config)
#        return env
#
#    return _thunk
#
#envs = [make_env(i) for i in range(1, num_envs)]
#envs = SubprocVecEnv(envs)

config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
          'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
          'allowed-modules': 0,
          'allowed-floors': 0,
          }

env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=1, docker_training=False, retro=True,
                       realtime_mode=False,
                       config=config)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = probs
        dist  = Categorical(logits=probs)
        return dist, value
  
def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
               
def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


num_inputs  = env.observation_space.shape[0]

policy_actions = [6, 12, 18, 3]
num_outputs = len(policy_actions)


#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4

#Threshold Rewards
threshold_reward = -200

max_frames = 15000
frame_idx  = 0
test_rewards = []

#State space flattened to get right shape and run - CNN would be preferable
state = env.reset()
state = state.flatten()
num_inputs  = len(state)#This will therefore have to change

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

early_stop = False

while frame_idx < max_frames and not early_stop:

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        ##Big Flag Here - the way the actions are selected and then stored as tensors for use in parameter updates needs to be checked 
        actioni = dist.sample()
        action = policy_actions[actioni]
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.flatten()

        log_prob = dist.log_prob(actioni)

        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        
        ###BIG FLAG HERE
        rewards.append(reward)#.to(device))
        #rewards.append(torch.FloatTensor(reward).to(device))
        #masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

        
        states.append(state)
        actions.append(actioni)
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            plot(frame_idx, test_rewards)
            if test_reward > threshold_reward: early_stop = True
            

    next_state = torch.FloatTensor(next_state).to(device)
    next_state = next_state.flatten()

    
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    
    #log_probs = torch.cat(log_probs).detach()
    log_probs = torch.stack(log_probs).detach()
    
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    
    #actions   = torch.cat(actions)
    actions   = torch.stack(actions)
    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
    
  
    
    
    
from itertools import count

max_expert_num = 50000
num_steps = 0
expert_traj = []

for i_episode in count():
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        expert_traj.append(np.hstack([state, action]))
        num_steps += 1
    
    print("episode:", i_episode, "reward:", total_reward)
    
    if num_steps >= max_expert_num:
        break
        
expert_traj = np.stack(expert_traj)
print()
print(expert_traj.shape)
print()
np.save("expert_traj.npy", expert_traj)
