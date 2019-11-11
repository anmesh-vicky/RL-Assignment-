import torch
import torch.nn as nn
import torch.nn.functional as F

class ANetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h1_size = 256, h2_size = 256):
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
        
        

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        pass
    
        
