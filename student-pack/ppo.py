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
from imitation_learner import unpickle_object, pickle_object
########################################################################
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):

    """
    Compute the returns of a trajectory.
    """
    values_temp = list(values)
    values_temp.append(next_value)
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        try:
            delta = rewards[step] + gamma * values_temp[step + 1] * masks[step] - values_temp[step]
        except:
            ipdb.set_trace()
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values_temp[step])
    return returns
########################################################################

def get_batch(mini_batch_size, states, actions, log_probs, values, returns, advantage):

    """
    Retrieve a mini-batch of data to do batch updates.
    """
    batch_size = states.size(0)

    for _ in range(batch_size // mini_batch_size):
        ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[ids, :], actions[ids], log_probs[ids], values[ids], returns[ids, :], advantage[ids, :]


########################################################################
def ppo_update(optimizer, model, ppo_epochs, mini_batch_size, states, actions, log_probs, values, returns, advantages, clip_param=0.2):

    """
    Update the ActorCritic model using the PPO algorithm.
    """

    batches_complete = 0
    while True:
        for state, action, old_log_probs, old_values, return_, advantage in get_batch(mini_batch_size, states, actions, log_probs, values, returns, advantages):

            outputs, value = model(state.reshape(-1, 3, 84, 336))
            probs = nn.Softmax(dim = -1)(outputs)
            dist  = Categorical(logits = probs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action.squeeze(1))

            ratio = (new_log_probs - old_log_probs.squeeze(1)).exp()


            surr1 = -advantage.squeeze(1) * ratio
            surr2 = -advantage.squeeze(1) * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

            clippedvalue = old_values + torch.clamp(value - old_values, -clip_param, clip_param)
            clippedvalue = (clippedvalue - return_).pow(2)

            unclippedvalue = (value - return_).pow(2)

            actor_loss  = torch.max(surr1, surr2).mean()
            critic_loss = 0.5 * torch.max(clippedvalue, unclippedvalue).mean()

            loss =  actor_loss + 0.75 * critic_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batches_complete += 1
        if batches_complete >= ppo_epochs:
            break
    try:
        return loss.detach().cpu().item(), optimizer
    except:
        ipdb.set_trace()

########################################################################

def train(env, model, gamma, max_epochs, batch_size, epochs_before_printing,
mini_batch_size, ppo_epochs, policy_actions, device, optimizer, max_steps_in_demo_episode,
show_demo, override_threshold, experiment):

    """
    Runs episodes to create trajectories, updates model with PPO.
    """

    hyper_params = {"learning_rate": 0.001, "steps": 1000000, "batch_size": 500}
    experiment.log_parameters(hyper_params)

    epoch = 0
    dequelen = 4
    steps = 0
    replay_buffer_length = 1000

    """"""
    log_probs = deque(maxlen = replay_buffer_length)
    values    = deque(maxlen = replay_buffer_length)
    states    = deque(maxlen = replay_buffer_length)
    actions   = deque(maxlen = replay_buffer_length)
    rewards   = deque(maxlen = replay_buffer_length)
    masks     = deque(maxlen = replay_buffer_length)
    """"""

    episode_hist = []

    statedeque = deque(maxlen = dequelen)
    nextstatedeque = deque(maxlen = dequelen)
    state = env.reset()
    statest = np.copy(state)

    for i in range(0, 4):
        statedeque.append(state) #initialising

    for i in range(0, 4):
        nextstatedeque.append(state)

    for frame in range(0, dequelen -1): #deque appends new frames to the right
        state = np.vstack((state, statedeque[frame])) #append all frames except the most recent

    while True:
        """
        log_probs = deque(maxlen = replay_buffer_length)
        values    = deque(maxlen = replay_buffer_length)
        states    = deque(maxlen = replay_buffer_length)
        actions   = deque(maxlen = replay_buffer_length)
        rewards   = deque(maxlen = replay_buffer_length)
        masks     = deque(maxlen = replay_buffer_length)
        """
        entropy = 0

        previous_time = 160 #for colour scoring
        state_memory = deque(maxlen = 4) #for unstuck agent
        state_memory.append(statest) #so that the memory has something

        #generate the trajectory
        for _ in range(batch_size):

            state_cpu = statest.copy()

            state = torch.FloatTensor(state).to(device).reshape(3, 84, 336).unsqueeze(0)

            outputs, value = model(state)
            probs = nn.Softmax(dim = -1)(outputs)
            dist  = Categorical(logits = probs)
            action = dist.sample()

            action = unstuck_agent(state_memory, state_cpu, action.item(), policy_actions, device, override_threshold = override_threshold) #unstuck the agent if stuck
            action = torch.Tensor(np.array([action])).to(device)

            env_action = policy_actions[int(action.detach().cpu().numpy())]
            next_state, reward, done, _ = env.step(env_action)

            statest = np.copy(next_state)

            nextstatedeque.append(next_state)

            for frame in range(0, dequelen - 1):
                next_state = np.vstack((next_state, nextstatedeque[frame]))

            if done: #tracking episode data
                episode_hist.append(steps)
                episodenumber = len(episode_hist)
                experiment.log_metric("Steps-per-episode", steps, step=episodenumber)
                print("Episode Number " +  str(episodenumber) + ": Steps to Termination: " +  str(steps))
                steps = 0
                done = False

            statest = np.copy(next_state)
            nextstatedeque.append(next_state)

            for frame in range(0, dequelen - 1):
                next_state = np.vstack((next_state, nextstatedeque[frame]))

            #for colour scoring - can comment this out to disable colour scoring
            previous_time, scores = colour_scoring(next_state, previous_time = previous_time)
            reward = reward * 10
            reward += scores.sum()

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            try:
                log_probs.append(log_prob.unsqueeze(0))
            except:
                ipdb.set_trace()
            values.append(value)
            rewards.append(reward)
            masks.append(1-done)
            states.append(state.flatten().unsqueeze(0))
            viewstate.append(statest)
            actions.append(action.unsqueeze(0))
            assert len(values) ==len(rewards)

            state_memory.append(state_cpu)
            state = next_state
            steps += 1

        next_state = torch.FloatTensor(next_state).to(device).reshape(3, 84, 336).unsqueeze(0)

        _, next_value = model(next_state)
        returns   = compute_gae(next_value, rewards, masks, values, gamma)
        returns   = torch.cat(returns).detach()
        returns   = (returns - returns.mean())/(returns.std() + 1e-4)

        log_probs_tensor = torch.cat(list(log_probs)).detach()
        values_tensor    = torch.cat(list(values)).detach()
        states_tensor    = torch.cat(list(states))
        actions_tensor   = torch.cat(list(actions))
        advantage = returns - values_tensor
        epoch += 1

        loss, optimizer = ppo_update(optimizer, model, ppo_epochs, mini_batch_size, states_tensor, actions_tensor, log_probs_tensor, values_tensor, returns, advantage)

        del log_probs_tensor, values_tensor, states_tensor, actions_tensor, advantage, returns


        #play an episode to see how we are doing
        if epoch%epochs_before_printing == 0:
            pickle_object(model, 'policy_model')
            rewardsum = np.sum(rewards)
            experiment.log_metric("Rewards-per-epoch", rewardsum, step=epoch)
            #experiment.log_asset("policy_model.pkl", overwrite = True)
            print("Number of epochs: " + str(epoch) + " | Reward: " + str(np.sum(rewards)) + " | loss " + str(loss))

        #return the model when done
        if epoch >= max_epochs:
            return model
