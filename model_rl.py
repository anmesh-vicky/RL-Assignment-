import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, state_size, action_size, seed = 123, h1_size = 64, h2_size = 64, lr=5e-4):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_size, h1_size)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(h1_size, h2_size)
        self.affine3 = nn.Linear(h2_size, action_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)

