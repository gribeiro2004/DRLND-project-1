#!/usr/bin/env python
# coding: utf-8

# ### 1. Fully connected network that maps both a state and an action onto a value

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, layer_sizes=[512, 128, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.layer_sizes = layer_sizes
        # Defining the layers
        self.fc1 = nn.Linear(self.state_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        # Output layer
        self.fc4 = nn.Linear(layer_sizes[2], action_size)
        
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        Q = self.fc1(state)
        Q = F.relu(Q)
        Q = self.fc2(Q)
        Q = F.relu(Q)
        Q = self.fc3(Q)
        Q = F.relu(Q)
        Q = self.fc4(Q)
        return Q

