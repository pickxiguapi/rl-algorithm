# -*- encoding: utf-8 -*-
"""
@File    : policy_gradient.py
@Time    : 2020/12/5
@Author  : Yuan Yifu
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import numpy as np

N_STATES = 4
N_ACTIONS = 2
GAMMA = 0.95
torch.manual_seed(2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        output = self.fc2(x)

        return F.softmax(output, dim=1)


class PolicyGradient(object):
    def __init__(self):
        self.policy_net = Net()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-2)

        self.obs = list()  # 单epi状态列表
        self.acs = list()  # 单epi动作列表
        self.rewards = list()  # 单epi奖励列表

    def choose_action(self, s):
        # 使用softmax生成的概率来选择动作

        s = torch.from_numpy(s).float()
        s = torch.FloatTensor(s).unsqueeze(0)
        prob = self.policy_net(s)

        # 创建以参数prob为标准的类别分布
        action = Categorical(prob).sample()

        return action.item()

    def store_transition(self, s, a, r):
        self.obs.append(s)
        self.acs.append(a)
        self.rewards.append(r)

    def learn(self):
        discounted_rs = self._discount_and_norm_rewards()

        # predict
        output = self.policy_net(torch.FloatTensor(self.obs))
        # one-hot code
        self.acs = torch.LongTensor(self.acs)
        temp_acs = self.acs.view(len(self.acs), 1)
        one_hot = torch.zeros(len(self.acs), N_ACTIONS).scatter_(1, temp_acs, 1)

        # to maximize total reward (log_p * R) is to minimize -(log_p * R)
        log_p = torch.sum(-torch.log(output)*one_hot, 1)
        loss = log_p * torch.Tensor(discounted_rs)
        loss = loss.mean()

        # optimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # finally, init it
        self.obs = list()  # 单epi状态列表
        self.acs = list()  # 单epi动作列表
        self.rewards = list()  # 单epi奖励列表

        return loss, discounted_rs

    def _discount_and_norm_rewards(self):
        # 定义非负数最小值，防止除数为0
        eps = np.finfo(np.float32).eps.item()

        discounted_rs = np.zeros_like(self.rewards)

        # calculate the true value using rewards returned from the env
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = self.rewards[t] + GAMMA * running_add
            discounted_rs[t] = running_add

        # normalize episode rewards
        discounted_rs -= np.mean(discounted_rs)
        discounted_rs /= (np.std(discounted_rs) + eps)

        return discounted_rs
