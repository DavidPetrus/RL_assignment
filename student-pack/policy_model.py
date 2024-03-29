import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import ipdb
from torch.utils.data import Dataset
from torch.distributions import Categorical

########################################################################
class PolicyModel(nn.Module):
    """
    NN Architecture that Learns our Policy.
    This is used for imitating behaviour from the recordings.
    """

    def __init__(self, action_size):
        """
        Basic CNN to learn the policy model.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(64 * 7 * 38, 512)
        self.linear2 = nn.Linear(512, action_size)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.view(-1, 64 * 7 * 38)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

########################################################################
class ActorCritic(nn.Module):
    """
    NN Architecture that Learns our Policy.
    This is used for imitating behaviour from the recordings.
    """

    def __init__(self, action_size):
        """
        Basic CNN to learn the policy model.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(64 * 7 * 38, 512)
        self.actor = nn.Linear(512, action_size)
        self.critic = nn.Linear(512, 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.view(-1,  64 * 7 * 38)
        x = self.linear1(x)
        value = self.critic(x)
        outputs = self.actor(x)#.sum(dim = 0)
        return outputs, value
########################################################################
