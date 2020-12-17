# -*- encoding: utf-8 -*-
"""
@File    : DDPG_net.py
@Time    : 2020/12/12 
@Author  : Yuan Yifu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fanin_init(size, fanin=None):
    """
    以合理的方式初始化weight
    :param size:
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v, v)  # 从-v到v的均匀分布
    return x.type(torch.FloatTensor)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(self.state_dim, 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(self.action_dim, 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        """
        根据状态和行为特征得到价值
        :param state: torch Tensor([n, state_dim])
        :param action: torch Tensor([n, action_dim])
        :return: Q(s, a) torch Tensor([n, 1])
        """
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)

        action = action.type(torch.FloatTensor)

        state = F.relu(self.fcs1(state))
        state = F.relu(self.fcs2(state))

        action = F.relu(self.fca1(action))

        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_domain):
        """

        :param state_dim: 状态特征数量
        :param action_dim: 动作特征数量
        :param action_domain: 动作取值区间[-action_domain, action_domain]
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_domain = action_domain

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(64, self.action_dim)
        self.fc4.weight.data.normal_(-0.003, 0.003)

    def forward(self, state):
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.tanh(self.fc4(x))  # (-1, 1)

        # implement output range
        action = action * self.action_domain

        return action
