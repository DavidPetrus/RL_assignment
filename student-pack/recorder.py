from RandomAgent import RandomAgent
from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import numpy as np
from PIL import Image
from utils.ffmpeg import export_video
import matplotlib.pyplot as plt


def big_obs(obs, info):
    """
    Big obs takes a retro observation and an info
    dictionary and produces a higher resolution
    observation with the retro features tacked on.
    """
    res = (info['brain_info'].visual_observations[0][0] * 255).astype(np.uint8)
    res[:20] = np.array(Image.fromarray(obs).resize((168, 168)))[:20]
    return res


def run_fn(env, agent):
    obs = env.reset()
    while True:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        yield big_obs(next_obs, info)
        if done:
            break
        obs = next_obs
    env.close()


if __name__ == '__main__':
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=1, docker_training=False, retro=True,
                           realtime_mode=False,
                           config=config)
    env.seed(1)
    agent = RandomAgent(env.observation_space, env.action_space)
    export_video('export_.mp4', 168, 168, 10, run_fn(env, agent))
