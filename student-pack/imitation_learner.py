import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import ipdb
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical
import pickle
from colour_segmentation import colour_scoring
from collections import deque
########################################################################
def verify_recording(play_recording = False):
    """
    Take the saved data and rebuild into images.
    Returns the images in shape (n, 3, 84, 84)
    """

    if play_recording:

        states = pd.read_csv('recordings/states.csv').values
        states = states.reshape(len(states), 84, 84, 3)
        actions = pd.read_csv('recordings/actions.csv').values
        rewards = pd.read_csv('recordings/rewards.csv').values
        keys = pd.read_csv('recordings/keys.csv').values

        print("Playing Video.")
        for state in range(states.shape[0]):
            plt.imshow(states[state,:,:,:])
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
########################################################################
def get_data():
    """
    Get the recordings and append the action (labels) for pytorch.
    """
    states = pd.read_csv('recordings/states.csv').values
    actions = pd.read_csv('recordings/actions.csv').values
    rewards = pd.read_csv('recordings/rewards.csv').values
    keys = pd.read_csv('recordings/keys.csv').values
    action_map, actions = map_new_labels(actions)
    data = np.hstack((states, actions))
    pickle_object(action_map, 'action_map')
    print("Retrieved Data.")
    return data, rewards, keys
########################################################################
def load_data(train, test, batch_size):
    """
    Get loaders for the data to train the nn.
    """
    train_loader = torch.utils.data.DataLoader(train, batch_size = 4, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = 4, shuffle = True)
    print("Loaded Data into DataLoaders..")
    return train_loader, test_loader
########################################################################
def map_new_labels(labels):
    """
    Remap the actions (18, 6...)
    To a range (0, 1...)
    So that we can use it with pytorch. :/
    """
    new_labels = labels.copy()
    map = {}
    unique_labels = np.unique(labels).tolist()
    freq_counts = np.array(np.unique(labels, return_counts=True)).T
    freq_counts[:,1] = freq_counts[:,1]*100/freq_counts[:,1].sum()
    print("The frequency of actions: ")
    print(freq_counts)
    counter = 0
    for label in unique_labels:
        new_labels[new_labels == label] = counter
        map[counter] = label
        counter += 1
    return map, new_labels
########################################################################
def train_test_split(data, train_fraction = 0.8):
    """
    split the data into train and test sets
    """
    np.random.shuffle(data)
    rows = data.shape[0]
    train_size = int(rows * train_fraction)
    train, test = data[: train_size,:], data[train_size:,:]

    print("The train set is " + str(len(train)) + " samples.")
    print("The test set is " + str(len(test)) + " samples.")

    train = DataObject(train)
    test = DataObject(test)
    return train, test
########################################################################
def train_model(policy_model, lr, train_loader, test_loader, device, num_epochs):
    """
    Train the CNN
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    optimizer = optim.Adam(policy_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    steps = 0

    print("Starting to learn policy...")

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()

            inputs, labels = data
            outputs, value = policy_model(inputs.float().to(device))
            probs = nn.Softmax(dim = -1)(outputs)
            dist  = Categorical(logits = probs)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            steps += 1

        if epoch % 100 == 0:
            print("epoch: " + str(epoch) + " loss " +  str(loss.item()))

    get_accuracy(policy_model, train_loader, device)
    get_accuracy(policy_model, test_loader, device)
    return policy_model
########################################################################
class DataObject(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image = self.data[index, :-1].reshape((3, 84, 84)).astype(np.uint8)
        label = self.data[index, -1]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
########################################################################
def get_accuracy(policy_model, data_loader_object, device):
    """
    Get the accuracy of the model.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader_object:
            images, labels = data
            outputs, value = policy_model(images.float().to(device))
            probs = nn.Softmax(dim = -1)(outputs)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print('Accuracy of the network images: %d %%' % (100 * correct / total))
########################################################################
def pickle_object(object_, name = 'policy_model'):
    """
    Pickle the policy object so that we can use it later.
    """
    file_name = name + ".pkl"
    filehandler = open(file_name,"wb")
    pickle.dump(object_,filehandler)
    filehandler.close()
    print("Pickling complete.")
########################################################################
def unpickle_object(name):
    """
    Unpickle the policy object.
    """
    file_name = name + ".pkl"
    file = open(file_name, 'rb')
    object_file = pickle.load(file)
    file.close()
    print("Retrieved Pickled Object.")
    return object_file
########################################################################
def imitation_play(env, policy_model, device, max_steps = 1000, override_threshold = 2000):
    """
    Let the agent play using the learnt policy_model.
    """
    action_map = unpickle_object('action_map')
    state = env.reset()
    done = False
    steps = 0
    previous_time = 0 #for colour scoring

    state_memory = deque(maxlen = 4)
    state = np.zeros((3, 84, 84))
    state_memory.append(state)

    while not done and steps < max_steps:

        outputs, value = policy_model(torch.Tensor(state.reshape(3, 84, 84)).unsqueeze(0).to(device)) #scores from linear layer in NN
        outputs += torch.Tensor(np.random.normal(0, 1, outputs.shape[0])).to(device) #add noise for exploration
        probs = nn.Softmax(dim = 1)(outputs) #convert outputs to probabilites
        dist = Categorical(probs) #make the policy a distribution
        unmapped_action = dist.sample() #get an action from the distribution
        #lookup the action so that the correct action is passed to unity
        action = unstuck_agent(state_memory, state.reshape(3, 84, 84), unmapped_action.item(), action_map, device, override_threshold = override_threshold) #unstuck the agent if stuck
        action = action_map[action]
        next_state, reward, done, info = env.step(action)

        previous_time, scores = colour_scoring(next_state, previous_time = previous_time) #for colour scoring
        #print((probs.detach().cpu().numpy() * 100).flatten().astype(np.uint8), unmapped_action)
        print(scores.flatten())

        plt.imshow(next_state)
        plt.draw()
        plt.pause(0.00001)
        plt.clf()

        state_memory.append(state.reshape(3, 84, 84))
        state = next_state
        steps+=1
########################################################################
def merge_recordings(write = False):
    """
    Since all of the data is stored seperately per level we need to combine the data.
    """

    if write: #only if we want to write

        import os

        actions = pd.DataFrame()
        states = pd.DataFrame()
        rewards = pd.DataFrame()
        keys = pd.DataFrame()

        for root, dirs, files in os.walk('recordings'):
            for file in files:

                if root != 'recordings': #dont append the master file to itself

                    data = pd.read_csv(os.path.join(root, file))

                    if file == 'actions.csv':
                        if len(actions) == 0:
                            actions = data.copy()
                        else:
                            actions = pd.concat((actions, data), sort = False, axis = 0)
                    elif file == 'states.csv':
                        if len(states) == 0:
                            states = data.copy()
                        else:
                            states = pd.concat((states, data), sort = False, axis = 0)
                    elif file == 'rewards.csv':
                        if len(rewards) == 0:
                            rewards = data.copy()
                        else:
                            rewards = pd.concat((rewards, data), sort = False, axis = 0)
                    elif file == 'keys.csv':
                        if len(keys) == 0:
                            keys = data.copy()
                        else:
                            keys = pd.concat((keys, data), sort = False, axis = 0)

        states.to_csv(os.path.join('recordings','states.csv'), index = False)
        actions.to_csv(os.path.join('recordings','actions.csv'), index = False)
        keys.to_csv(os.path.join('recordings','keys.csv'), index = False)
        rewards.to_csv(os.path.join('recordings','rewards.csv'), index = False)
        print("Created master recording file.")
########################################################################
def unstuck_agent(state_memory, state, action, action_map, device, override_threshold = 100):
    """
    Force more exploration when the agent is stuck.
    Iterate over the frames in the frame memory and determine how much the state has changed.
    If there is a small change then we random action.
    Override threshold is per frame in the state memory
    """

    frame_similarity = 0
    for frame in state_memory:
        frame_similarity += np.abs(frame - state).sum()

    if frame_similarity < override_threshold * len(state_memory): #if stuck the frame difference will be very similar
        return np.random.randint(0, len(action_map.keys())) #choose random action

    return action

########################################################################
