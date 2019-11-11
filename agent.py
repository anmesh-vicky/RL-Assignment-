import numpy as np
import random
from collections import namedtuple, deque

from model import ANetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 256       # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # A-Network
        ## TODO: Initialize your action network here
        "*** YOUR CODE HERE ***"

        # A-Network
        self.qnetwork_local = ANetwork(state_size, action_size, seed).to(device)
#         self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, log_prob):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, log_prob)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            highest_prob_action, log_prob = self.qnetwork_local.get_action(state)
        self.qnetwork_local.train()
#         print(log_prob)
        return highest_prob_action, log_prob

        # Epsilon-greedy action selection
#         if random.random() > eps:
#             return np.argmax(action_values.cpu().data.numpy())
#         else:
#             return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma = GAMMA):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, log_probs = experiences

        ## TODO: compute and minimize the loss using REINFORCE
        "*** YOUR CODE HERE ***"
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        discounted_rewards = discounted_rewards.to('cuda')
        policy_gradient = []
#         print(log_probs)
#         print(discounted_rewards)
#         print()
#         print(discounted_rewards.shape, log_probs.shape)
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.qnetwork_local.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient = Variable(policy_gradient, requires_grad = True)
        policy_gradient.backward()
        self.qnetwork_local.optimizer.step()
        
                             


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): Total number of actions
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "log_prob"])
        self.seed = random.seed(seed)
        self.curr = 0
    
    def add(self, state, action, reward, next_state, done, log_prob):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, log_prob)
        self.memory.append(e)
        
    def clear(self):
        self.memory = deque(maxlen=BUFFER_SIZE)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = []
#         print("len is " + str(len(self.memory)))
        for i in range(self.batch_size):
#             print(self.curr)
            experiences.append(self.memory[self.curr])    
            self.curr += 1
            if self.curr >= len(self.memory):
                self.curr = 0

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        log_probs = torch.from_numpy(np.vstack([e.log_prob for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, log_probs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)