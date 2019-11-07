########################################################################
import gym
import ipdb
import matplotlib.pyplot as plt
from collections import deque
import keyboard
import time
import datetime
import random
import numpy as np
from recorder import create_env, record_game, save_file_as_csv
########################################################################
"""
https://discourse.aicrowd.com/t/actions-meaning-in-retro-mode/931
6 - rotate counter-clockwise
12 - rotate clockwise
18 - move forward
3 - jump
22 - jump and move forward - may want to include this
"""

action_map = {'w': 18, 'q': 6, 'e': 12, 'a': 3, 's': 21}
max_steps_to_record = 10e100
starting_floor = 0
total_floors = 10

########################################################################
env = create_env(starting_floor = starting_floor, total_floors = total_floors)
state_hist, action_hist, reward_hist, key_hist =  record_game(env, max_steps_to_record, action_map)
save_file_as_csv(state_hist, action_hist, reward_hist, key_hist)
########################################################################
