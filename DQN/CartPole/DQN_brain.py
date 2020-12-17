# -*- encoding: utf-8 -*-
"""
@File    : DQN_brain.py
@Time    : 2020/12/4 
@Author  : Yuan Yifu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
MEMORY_CAPACITY = 2000
N_STATE = 4
N_ACTIONS = 2
EPSILON = 0.9
LR = 0.01
REPLACE_COUNTER = 100
BATCH_SIZE = 32
GAMMA = 0.9


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATE, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)

        return x


class DQN(object):
    def __init__(self):
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATE*2+2))
        self.memory_counter = 0
        self.learn_step_counter = 0

        # init Net
        self.eval_net, self.target_net = Net(), Net()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_function = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target replace
        if self.learn_step_counter % REPLACE_COUNTER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(batch_memory[:, :N_STATE])
        b_a = torch.LongTensor(batch_memory[:, N_STATE:N_STATE+1].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, N_STATE+1:N_STATE+2])
        b_s_ = torch.FloatTensor(batch_memory[:, -N_STATE:])

        # q_eval
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch, 1)

        # q_target
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * torch.max(q_next, 1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, s):
        # e.g. s = [5, 4, 8, 7, 4] → [[]]
        s = torch.FloatTensor(s)
        s = torch.unsqueeze(s, 0)

        # epsilon greedy
        if np.random.uniform() < EPSILON:
            # greedy
            actions_value = self.eval_net(s)
            action = torch.max(actions_value, 1)[1].numpy()
            action = action[0]
        else:
            # random
            action = np.random.randint(0, N_ACTIONS)

        return action
