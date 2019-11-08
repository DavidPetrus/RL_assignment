import numpy as np
import pandas as pd
from imitation_learner import *
from policy_model import ActorCritic
import ipdb
import torch

def main():
########################################################################

    """
    Setup the meta params.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print("The device is " + str(device))

    build_master_file = False #compiles data from each level
    play_recording = False #plays video of all states
    batch_size = 32 #batch_size of data in trainging
    train_fraction = 0.7 #proportion of data used for training
    lr = 0.001 #learning rate of the model
    num_epochs = 50 #number of times data is used in training

########################################################################

    """
    Recover the data from the different levels
    """

    merge_recordings(write = build_master_file)
    verify_recording(play_recording = play_recording) #plays video of recording

########################################################################

    data, rewards, keys = get_data() #retrieve the data
    train, test = train_test_split(data, train_fraction = train_fraction) #train test split
    train_loader, test_loader = load_data(train, test, batch_size) #convert to loader objects

########################################################################

    policy_model = ActorCritic(4).to(device) #initialize the policy object
    policy_model = train_model(policy_model, lr, train_loader, test_loader, device, num_epochs = num_epochs) #train the model

########################################################################

    pickle_object(policy_model) #save the object to disk

########################################################################
if __name__ == "__main__":
    main()
