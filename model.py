import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.distributions import Categorical


class ANetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h1_size = 64, h2_size = 64, lr=5e-4):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Total number of actions
            seed (int): Random seed
        """
        super(ANetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.num_actions = action_size
        self.linear1 = nn.Linear(state_size, h1_size)
        self.linear2 = nn.Linear(h1_size, h2_size)
        self.linear3 = nn.Linear(h2_size, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        
        

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)
        return x
    
    def get_action(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
#         highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.cpu().data.numpy()))
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
#         log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
#         return highest_prob_action, log_prob.to('cuda')
    
        
