########################################################################
import math
import random
import ipdb
import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import ipdb
from torch.distributions import Categorical
from utils import im2vid, change_resolution
from imitation_learner import unstuck_agent
from colour_segmentation import colour_scoring
########################################################################
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):

    """
    Compute the returns of a trajectory.
    """

    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

########################################################################

def get_batch(mini_batch_size, states, actions, log_probs, returns, advantage):

    """
    Retrieve a mini-batch of data to do batch updates.
    """

    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[ids, :], actions[ids], log_probs[ids], returns[ids, :], advantage[ids, :]

########################################################################

def ppo_update(optimizer, model, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):

    """
    Update the ActorCritic model using the PPO algorithm.
    """

    batches_complete = 0
    while True:
        for state, action, old_log_probs, return_, advantage in get_batch(mini_batch_size, states, actions, log_probs, returns, advantages):

            outputs, value = model(state.reshape(-1, 3, 84, 84))
            probs = nn.Softmax(dim = -1)(outputs)
            dist  = Categorical(logits = probs)
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

        batches_complete += 1
        if batches_complete >= ppo_epochs:
            break

    return loss.detach().cpu().item(), optimizer

########################################################################

def train(env, model, gamma, max_epochs, batch_size, epochs_before_printing,
mini_batch_size, ppo_epochs, policy_actions, device, optimizer, max_steps_in_demo_episode,
show_demo, override_threshold):

    """
    Runs episodes to create trajectories, updates model with PPO.
    """

    state = env.reset()
    epoch = 0

    while True:

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy   = 0

        previous_time = 0 #for colour scoring
        state_memory = deque(maxlen = 4) #for unstuck agent
        state_memory.append(state) #so that the memory has something

        #generate the trajectory
        for _ in range(batch_size):
            state_cpu = state.copy()
            state = torch.FloatTensor(state).to(device).reshape(3, 84, 84).unsqueeze(0)
            outputs, value = model(state)
            probs = nn.Softmax(dim = -1)(outputs)
            dist  = Categorical(logits = probs)
            action = dist.sample()

            action = unstuck_agent(state_memory, state_cpu, action.item(), policy_actions, device, override_threshold = override_threshold) #unstuck the agent if stuck
            action = torch.Tensor(np.array([action])).to(device)

            env_action = policy_actions[int(action.detach().cpu().numpy())]
            next_state, reward, done, _ = env.step(env_action)

            #for colour scoring - can comment this out to disable colour scoring
            previous_time, scores = colour_scoring(next_state, previous_time = previous_time)
            # if scores.sum() > 0:
            #     print((scores * 100000000).astype(np.uint8))
            #     plt.imshow(next_state)
            #     plt.show()
            reward += scores.sum()

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob.unsqueeze(0))
            values.append(value)
            rewards.append(reward)
            masks.append(1-done)
            states.append(state.flatten().unsqueeze(0))
            actions.append(action.unsqueeze(0))

            state_memory.append(state_cpu)
            state = next_state

        #calculate the return
        next_state = torch.FloatTensor(next_state).to(device).reshape(3, 84, 84).unsqueeze(0)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values, gamma)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        epoch += 1

        loss, optimizer = ppo_update(optimizer, model, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

        #play an episode to see how we are doing
        if epoch%epochs_before_printing == 0:
            print("Number of epochs: " + str(epoch) + " | Reward: " + str(np.sum(rewards)) + " | loss " + str(loss))
            if show_demo:
                im2vid(run_episode(model, env, policy_actions, max_steps_in_demo_episode, show_demo, device), 'video ' + str(epoch) + ' epochs')
                #state = env.reset()

        #return the model when done
        if epoch >= max_epochs:
            return model

########################################################################

def run_episode(model, env, policy_actions, max_steps_in_demo_episode, show_demo, device):

    """
    Run an episode in order to test how well we are doing.

    max_steps_in_episode: the number of steps we allow in 1 episode.
    """

    steps = 0
    state = env.reset()
    done = False
    total_reward = 0

    while True:
        state = torch.FloatTensor(state).reshape(3, 84, 84).unsqueeze(0).to(device)
        outputs, value = model(state)
        probs = nn.Softmax(dim = -1)(outputs)
        dist  = Categorical(logits = probs)
        action = dist.sample()
        next_state, reward, done, info = env.step(policy_actions[int(action.detach().cpu().numpy())])
        if show_demo:
            yield change_resolution(next_state, info)
        state = next_state
        total_reward += reward
        steps += 1

        if steps >= max_steps_in_demo_episode or done:
            break

########################################################################
