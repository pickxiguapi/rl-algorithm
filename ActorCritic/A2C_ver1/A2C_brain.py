# -*- encoding: utf-8 -*-
"""
@File    : A2C_brain.py
@Time    : 2020/12/6 
@Author  : Yuan Yifu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
N_STATES = 4
N_ACTIONS = 2
GAMMA = 0.95
LR = 0.01


class Net(nn.Module):
    """
    implements both actor and critic in one model
    使用一个网络来拟合 actor 和 critic
    """

    def __init__(self):
        super(Net, self).__init__()
        self.affine1 = nn.Linear(N_STATES, 128)

        # actor's layer
        self.action_head = nn.Linear(128, N_ACTIONS)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choose action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t     V(s)
        return action_prob, state_values


class A2C(object):
    def __init__(self):
        self.model = Net()
        self.SavedAction = namedtuple("SavedAction", ['log_prob', 'value'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

    def add_rewards(self, r):
        self.model.rewards.append(r)

    def choose_action(self, state):
        # numpy to torch
        state = torch.from_numpy(state).float()

        # the forward propagation
        prob, value = self.model(state)
        # sample prob
        m = torch.distributions.Categorical(prob)

        # and sample an action using the distribution
        action = m.sample()
        # save to action buffer
        # log p, state_value
        self.model.saved_actions.append(self.SavedAction(m.log_prob(action), value))

        return action.item()

    def learn(self):
        """
        calculate actor and critic loss and back propagation.
        :return:
        """
        discounted_rs = self._discount_and_norm_rewards(self.model.rewards)
        saved_action = self.model.saved_actions

        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss

        for (log_p, value), r in zip(saved_action, discounted_rs):
            # 核心算法

            # advantage baseline
            advantage = r - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_p * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).float()))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def _discount_and_norm_rewards(self, rewards):
        # 定义非负数最小值，防止除数为0
        eps = np.finfo(np.float32).eps.item()

        discounted_rs = np.zeros_like(rewards)

        # calculate the true value using rewards returned from the env
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = rewards[t] + GAMMA * running_add
            discounted_rs[t] = running_add

        # normalize episode rewards
        discounted_rs -= np.mean(discounted_rs)
        discounted_rs /= (np.std(discounted_rs) + eps)

        return discounted_rs
