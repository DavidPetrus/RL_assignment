from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation

import numpy as np
import sys


def run_episode(env):
    # create instance of MyAgent
    from MyAgent import MyAgent
    agent = MyAgent(env.observation_space, env.action_space)

    done = False
    episode_return = 0.0
    state = env.reset()
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        episode_return += reward
        state = new_state
    return episode_return


if __name__ == '__main__':
    error_occurred = False
    # In this example we use the seeds used for evaluating submissions
    # to the Obstacle Tower Challenge.
    eval_seeds = [1, 2, 3, 4, 5]

    # Create the ObstacleTowerEnv gym and launch ObstacleTower
    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    worker_id = int(np.random.randint(999, size=1))
    print(worker_id)
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', docker_training=False, worker_id=worker_id, retro=True,
                           realtime_mode=False, config=config)

    # Wrap the environment with the ObstacleTowerEvaluation wrapper
    # and provide evaluation seeds.
    env = ObstacleTowerEvaluation(env, eval_seeds)

    # We can run episodes (in this case with a random policy) until
    # the "evaluation_complete" flag is True.  Attempting to step or reset after
    # all of the evaluation seeds have completed will result in an exception.

    while not env.evaluation_complete:
        try:
            episode_rew = run_episode(env)
        except Exception as exception:
            with open('error_evaluation.txt', 'a') as error_file:
                error_file.write('\n' + str(exception.msg) + '\n')
                error_occurred = True
            break

    # Finally the evaluation results can be fetched as a dictionary from the
    # environment wrapper.
    env.close()
    if error_occurred:
        print(-100.0)
    else:
        print(env.results['average_reward']*10000)
