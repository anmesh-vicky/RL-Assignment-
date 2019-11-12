import numpy as np
from itertools import count
from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


LR = 5e-4
BATCH_SIZE = 64
GAMMA = 0.99

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", no_graphics=True)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)


policy = Policy(state_size, action_size)
policy.load_state_dict(torch.load('checkpoint_re.pth'))
policy.train()
optimizer = optim.Adam(policy.parameters(), lr=LR)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    scores_window = deque(maxlen=100)
    max_score = -1
    for i_episode in range(1, 4001):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        ep_reward = 0
        for t in range(1, 1000):  # Don't infinite loop while learning
            action = select_action(state)
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]            
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        scores_window.append(ep_reward)
        finish_episode()
        print("Episode Reward {} : {:.2f}".format(i_episode, ep_reward))
        if i_episode % 100 == 0:
            cur = np.mean(scores_window)
            
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, cur))
            
            torch.save(policy.state_dict(), 'checkpoint_' + str(i_episode) + str(cur) + '.pth')
            
            if cur >= 13:
                torch.save(policy.state_dict(), 'checkpoint_final' + str(i_episode) + str(cur) + '.pth')
            
            if cur > max_score:
                max_score = cur
                torch.save(policy.state_dict(), 'checkpoint_re.pth')


if __name__ == '__main__':
    main()