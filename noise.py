'''OUNoise base class for initialization of noise object.
Since, the output of the actor network (in ddpg) is deterministic, we add noise to the output(continuous action)
to facilitate the exploration for the agent.

@author: Rohith Banka.
'''

import random
import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        '''Initializes the OUNoise object.

        Params:
        ======
            size (int): size of the action tensor.
            seed (int): random seed to preserve the configuration.
            mu  (float): Mean of the OUNoise process.
            theta
            sigma'''
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.size = size

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state