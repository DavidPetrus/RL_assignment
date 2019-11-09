########################################################################
import gym
import ipdb
import matplotlib.pyplot as plt
import keyboard
import time
import datetime
import random
import numpy as np
import pandas as pd
from colour_segmentation import colour_scoring
from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
########################################################################
def record_game(env, max_steps_to_record, action_map, stroke_time_diff = 0.3):

    """
    Set matplotlib hotkeys off.
    This stops the plot from doing funny things.
    """

    plt.rcParams['keymap.save'] = ''
    plt.rcParams['keymap.pan'] = ''
    plt.rcParams['keymap.all_axes'] = ''
    plt.rcParams['keymap.quit'] = ''

    """
    Initialise some variables.
    """

    pause_key = "p" #switches between recording and non recording mode
    record_mode = True #switch for recording mode
    state_hist = []
    action_hist = []
    reward_hist = []
    key_hist = []
    state = env.reset()
    steps = 0
    time_of_previous_press = datetime.datetime.now()
    done = False
    previous_time = 0.1
    print("Waiting for first action")

    while True:

        """
        Get the key stroke from the player.
        """

        action = keyboard.read_key() #get the key stroke

        """
        Capture the time of the stroke. This is used to make sure we dont accidently push a key twice (hold it in).
        We only keep key strokes if they are at least stroke_time_diff apart.
        """

        time_of_press = datetime.datetime.now() #record the time of the press
        time_diff = (time_of_press - time_of_previous_press).total_seconds() #record the time difference

        """
        Frist do some tests on the key stroke to make sure it will not crash the code.
        Also handles pause, and quit operations.
        """

        if action == 'esc': #test if we want to quit and return the frames that we have so far
            env = close_sequence(action_hist, state_hist, reward_hist, env)
            return state_hist, action_hist, reward_hist, key_hist
            break
        if action not in action_map.keys() and action != pause_key: #catch cases where you push a wrong key by mistake
            print("pressed non-action key" + str(action))
            continue
        if action == pause_key and time_diff > stroke_time_diff:
            if record_mode:
                record_mode = False
                time_of_previous_press = time_of_press
                print("Now switching off record mode.")
            else:
                record_mode = True
                time_of_previous_press = time_of_press
                print("Now switching on record mode.")
            continue #go to next loop since we cant take this action
        elif action == pause_key and time_diff <= stroke_time_diff:
            continue
        else:
            action = action_map[action] #get the action from our map

        """
        If it is a valid action then we take the action and record the states.
        """

        if time_diff > stroke_time_diff: #only capture if there was a small gap of time between actions

            #take the step
            next_state, reward, done, info = env.step(action)

            #get the colour scores
            previous_time, scores = colour_scoring(next_state, previous_time = previous_time) #for colour scoring

            #record the state and action
            if record_mode: #only if record mode is on
                state_hist.append(state.flatten())
                action_hist.append(action)
                reward_hist.append(reward)
                key_hist.append(info['total_keys'])

            #show the state
            plt.clf()
            plt.imshow(next_state)
            plt.draw()
            plt.pause(0.0001)

            #update vars
            time_of_previous_press = time_of_press
            state = next_state
            print("Action Taken " + str(action) + ", reward: " +str(reward)  + " total_keys: " + str(info['total_keys']) + " current floor " + str(info['current_floor']))

        if done:
            env = close_sequence(action_hist, state_hist, reward_hist, env)
            return state_hist, action_hist, reward_hist, key_hist

########################################################################
def close_sequence(action_hist, state_hist, reward_hist, env):
    """
    Final sequence when end of recording session
    """
    print("the actions are " + str(len(action_hist)) + " and " + "the states are " + str(len(state_hist)) + " rewards " + str(len(reward_hist)))
    env.close()
    print('Finished Recording Session')
    return env
########################################################################
def save_file_as_csv(states, actions, rewards, keys):
    """
    Save a recording as csv.
    Flattens each image (state).
    """
    states = np.array((states))
    actions = np.array((actions))
    rewards = np.array((rewards))
    keys = np.array((keys))

    states = pd.DataFrame(states)
    states.to_csv('recordings/states.csv', index = False)
    actions = pd.DataFrame(actions)
    actions.to_csv('recordings/actions.csv', index = False)
    rewards = pd.DataFrame(rewards)
    rewards.to_csv('recordings/rewards.csv', index = False)
    keys = pd.DataFrame(keys)
    keys.to_csv('recordings/keys.csv', index = False)

    print("The files were saved to csv")
########################################################################
def create_env(starting_floor = 0, total_floors = 10, worker_id = 1):
    """
    Here we set up the environement according to the assignment instructions.
    The total floors is update by one if equal to starting floor.
    """

    assert starting_floor < total_floors, "Invalid Floors Specified Start: " + str(starting_floor) + " total: " + str(total_floors)


    config = {'starting-floor': starting_floor, 'total-floors': total_floors, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              'worker_id': worker_id
              }
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', worker_id=1, docker_training=False, retro=True,realtime_mode=False,config=config)
    env.seed(1)
    #_ = env.reset()
    return env
########################################################################
