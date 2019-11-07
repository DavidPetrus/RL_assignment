########################################################################
from ppo import train
from dummy_env import DummyEnv
from recorder import create_env
import torch
import torch.optim as optim
from policy_model import ActorCritic
########################################################################
"""
Setup parameters.
"""

#learning parameters
gamma = 0.99
learning_rate = 0.001

#batch update parameters
max_epochs = 10 #number of batches to create
batch_size = 100 #number of steps in one batch
mini_batch_size = 5 #how many steps are used to update the loss
ppo_epochs = 20 #number of batches to sample in ppo update || min_batch * ppo_epoch = batch_size recommended
epochs_before_printing = 5 #number of steps at which we print update

#demo information
show_demo = True #whether or not to make a video of the current progress
max_steps_in_demo_episode = 200 #number of steps to show in demo episode

#environement parameters
starting_floor = 0
total_floors = 1
env = create_env(starting_floor, total_floors)
#env = DummyEnv()
policy_actions = [6, 12, 18, 21] #allowed actions by agent

#deep learning setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(len(policy_actions)).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
########################################################################
"""
Train the model.
"""
model = train(env, model, gamma, max_epochs, batch_size, epochs_before_printing,
mini_batch_size, ppo_epochs, policy_actions, device, optimizer, max_steps_in_demo_episode, show_demo)
########################################################################
