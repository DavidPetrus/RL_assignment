from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt


def main():
    config = {'starting-floor': 0, 'total-floors': 5, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower',
                           worker_id=1, retro=True, realtime_mode=False, config=config)
    env.seed(1)
    print(env.observation_space)
    print(env.action_space)

    obs = env.reset()

    plt.imshow(obs)
    plt.show()

    obs, reward, done, info = env.step(env.action_space.sample())
    print('obs', obs)
    print('reward', reward)
    print('done', done)
    print('info', info)

    plt.imshow(obs)
    plt.show()
    env.close()


if __name__ == '__main__':
    main()
