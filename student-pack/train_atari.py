import argparse
import random
import torch
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from ppo import train
from dummy_env import DummyEnv
from recorder import create_env
import torch
import torch.optim as optim
from policy_model import ActorCritic
from imitation_learner import unpickle_object
import ipdb
from utils import im2vid, change_resolution

if __name__ == '__main__':
    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay_buffer_size": int(5e3),  # replay buffer size
        "learning_rate": 1e-4,  # learning rate for Adam optimizer
        "discount_factor": 0.99,  # discount factor
        "num_steps": 50000,  # total number of steps to run the environment for
        "batch_size": 32,  # number of transitions to optimize at the same time
        "learning_starts": 10000,  # number of steps before learning starts
        "learning_freq": 1,  # number of iterations between every optimization step
        "use_double_dqn": False,  # use double deep Q-learning
        "target_update_freq": 1000,  # number of iterations between every target network update
        "eps_start": 1.0,  # e-greedy start threshold
        "eps_end": 0.01,  # e-greedy end threshold
        "eps_fraction": 0.1,  # fraction of num-steps
        "print_freq": 10
    }

    #====
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
            action = agent.act(state.squeeze(0).detach().cpu().numpy())
            next_state, reward, done, info = env.step(policy_actions[action])


            if show_demo:
                yield change_resolution(next_state, info)
            state = next_state
            total_reward += reward
            steps += 1

            if steps >= max_steps_in_demo_episode or done:
                break
    #====
    torch.manual_seed(hyper_params['seed'])
    torch.cuda.manual_seed_all(hyper_params['seed'])
    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])

    assert "NoFrameskip" in hyper_params['env'], "Require environment with no frameskip"
    env = create_env(0,1)
    env.seed(hyper_params['seed'])
    #env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    #env = EpisodicLifeEnv(env)
    #env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 3)

    replay_buffer = ReplayBuffer(hyper_params['replay_buffer_size'])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params['use_double_dqn'],
        lr=hyper_params['learning_rate'],
        batch_size=hyper_params['batch_size'],
        gamma=hyper_params['discount_factor']
    )

    eps_timesteps = hyper_params['eps_fraction'] * float(hyper_params['num_steps'])
    episode_rewards = [0.0]
    loss = [0.0]
    policy_actions = unpickle_object('action_map')



    state = env.reset()
    for t in range(hyper_params['num_steps']):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params['eps_start'] + fraction * (hyper_params['eps_end'] - hyper_params['eps_start'])
        sample = random.random()
        if sample > eps_threshold:
            action = agent.act(np.array(state))
        else:
            action = np.random.randint(0,4)

        env_action = policy_actions[action]
        next_state, reward, done, _ = env.step(env_action)
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if t > hyper_params['learning_starts'] and t % hyper_params['learning_freq'] == 0:
            agent.optimise_td_loss()

        if t > hyper_params['learning_starts'] and t % hyper_params['target_update_freq'] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params['print_freq'] is not None and len(episode_rewards) % hyper_params['print_freq'] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            im2vid(run_episode(agent.policy_network, env, policy_actions, 200, True, 'cuda'), 'video ' + str(num_episodes) + ' episodes')

import ipdb
ipdb.set_trace()
