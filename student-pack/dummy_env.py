import torch
import numpy as np

########################################################################
class DummyEnv():

    """
    Create a dummy env that we can use while we debug.

    The only point of this is so that we dont have to wait for the unity environemnt to start every time.
    """

    def __init__(self):
        self.a = 123

    def step(self, action):

        next_state = np.random.randint(255, size = (84, 84, 3))
        reward = -1
        done = False
        info = ''
        return next_state, reward, done, info

    def reset(self):
        return np.random.randint(255, size = (84, 84, 3))

########################################################################
