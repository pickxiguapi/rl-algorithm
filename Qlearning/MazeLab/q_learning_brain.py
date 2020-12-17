# -*- encoding: utf-8 -*-
"""
@File    : sarsa_lambda_brain.py
@Time    : 2020/11/29 
@Author  : Yuan Yifu
"""
import numpy as np
import pandas as pd


ALPHA = 0.01
GAMMA = 0.9
EPSILON = 0.9


class Qlearning(object):
    def __init__(self, action_space, learning_rate=ALPHA, reward_decay=GAMMA, e_greedy=EPSILON):
        self.actions = action_space  # 动作空间
        self.lr = learning_rate  # 学习率alpha
        self.gamma = reward_decay  # 奖励衰减gamma
        self.epsilon = e_greedy  # epsilon greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # add this state to the data frame
            self.q_table = self.q_table.append(
                pd.Series(data=[0]*len(self.actions),
                          index=self.q_table.columns,
                          name=state,
                          ))
        else:
            pass

    def choose_action(self, observation):
        self.check_state_exist(observation)

        # epsilon greedy
        if np.random.random() < self.epsilon:
            # choose best action
            action_q_value_list = self.q_table.loc[observation, :]
            # maybe the same value
            action = np.random.choice(action_q_value_list[action_q_value_list == np.max(action_q_value_list)].index)
        else:
            # random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, observation, a, r, observation_):
        # judge next state exist
        self.check_state_exist(observation_)

        # choose best action
        # choose best action
        action_q_value_list = self.q_table.loc[observation_, :]
        # maybe the same value
        max_action = np.random.choice(action_q_value_list[action_q_value_list == np.max(action_q_value_list)].index)

        if observation_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[observation_, max_action]
        else:
            q_target = r

        q_predict = self.q_table.loc[observation, a]

        self.q_table.loc[observation, a] += self.lr * (q_target - q_predict)
