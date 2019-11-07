import numpy as np
import pandas as pd
from imitation_learner import *
from policy_model import PolicyModel
from recorder import create_env
import ipdb
import torch

def main():
########################################################################

    """
    Initialize variables.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print("The device is " + str(device))
    starting_floor = 0
    total_floors = 9
    max_steps = 100000
    override_threshold = 1000

########################################################################

    """
    Retreive the learnt policy.
    """

    policy_model = unpickle_object("policy_model") #read the object from disk

########################################################################

    """
    Let the agent play a game using the policy.
    """

    env = create_env(starting_floor = starting_floor, total_floors = total_floors)
    imitation_play(env, policy_model, device, max_steps = max_steps, override_threshold = override_threshold)

########################################################################
if __name__ == "__main__":
    main()
