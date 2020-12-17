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
TRACE = 0.9


class SarsaLambda(object):
    def __init__(self, action_space, learning_rate=ALPHA, reward_decay=GAMMA, e_greedy=EPSILON, trace_decay=TRACE):
        self.actions = action_space  # 动作空间
        self.lr = learning_rate  # 学习率alpha
        self.gamma = reward_decay  # 奖励衰减gamma
        self.epsilon = e_greedy  # epsilon greedy
        self.lambda_ = trace_decay

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.eligibility_traces = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # add this state to the data frame
            self.q_table = self.q_table.append(
                pd.Series(data=[0]*len(self.actions),
                          index=self.q_table.columns,
                          name=state,
                          ))
            self.eligibility_traces = self.eligibility_traces.append(
                pd.Series(data=[0] * len(self.actions),
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

    def init_eligibility_traces(self):
        self.eligibility_traces *= 0

    def learn(self, s, a, r, s_, a_):
        """

        Sarsa(lambda) 核心改动
        """
        # judge next state exist
        self.check_state_exist(s_)
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        q_predict = self.q_table.loc[s, a]
        error = q_target - q_predict

        # Method 2:
        self.eligibility_traces.loc[s, :] *= 0
        self.eligibility_traces.loc[s, a] = 1

        # Q UPDATE
        self.q_table += self.lr * error * self.eligibility_traces

        # decay eligibility trace after update
        self.eligibility_traces *= self.gamma * self.lambda_