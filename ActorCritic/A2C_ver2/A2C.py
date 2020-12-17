# -*- encoding: utf-8 -*-
"""
Advantage Actor Critic(A2C)
AC policy loss: -log_prob(a) * Q(s, a)
A2C policy loss: -log_prob(a) * (Q(s, a) - V(s)) V(s') replaced by R(s')
@File    : A2C.py
@Time    : 2020/12/8 
@Author  : Yuan Yifu
"""
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        # action & reward buffer
        self.saved_log_prob = []


ENV = 'CartPole-v0'
DISCRETE = True
HIDDEN_DIM = 30
env = gym.make('CartPole-v0')
env = env.unwrapped
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

actor_net = ActorNetwork().to(device)
critic_net = CriticNetwork().to(device)


actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-2)


def train():
    # hyper-parameters
    max_episodes = 3000
    max_steps = 1000  # short time step would be too easy for CartPole

    frame_idx = 0
    running_reward = 10
    episode_rewards = []

    for i_episode in range(max_episodes):
        state = env.reset()

        episode_reward = 0
        if ON_POLICY:
            rewards = []
        if not DETERMINISTIC:
            entropies = 0
        for step in range(max_steps):
            frame_idx += 1
            if ON_POLICY:
                action, log_prob, entropy = actor_net.evaluate_action(state)
                # print('state: ', state, 'action: ', action, 'log_prob: ', log_prob)
                state_value = critic_net(state)

                if ENV == 'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                # elif ENV ==  'Pendulum':
                else:  # gym env
                    if DISCRETE:
                        next_state, reward, done, _ = env.step(action[0])  # discrete action only needs a index
                    else:
                        next_state, reward, done, _ = env.step(action)
                    env.render()
                next_state_value = critic_net(next_state)
                actor_net.saved_entropies.append(entropy)
                if UPDATE == 'Approach0':
                    # this is critical! have to save the values inside the model to keep track of its gradients
                    actor_net.saved_logprobs.append(log_prob)
                    critic_net.saved_values.append(state_value)
                    # SavedSet.append(SavedTuple(log_prob, state_value))
                if UPDATE == 'Approach1':
                    # this is critical! have to save the values inside the model to keep track of its gradients
                    actor_net.saved_logprobs.append(log_prob)
                    critic_net.saved_values.append(state_value)
                    critic_net.saved_nextvalues.append(next_state_value)
                    # SavedSet.append(SavedTuple2(log_prob, state_value, next_state_value))
                if done:
                    reward = -20 if ENV == 'CartPole-v0' else reward
                    break
                rewards.append(reward)
            else:  # off-policy update with memory buffer
                pass
                if done:
                    reward = -20 if ENV == 'CartPole-v0' else reward
                    break

            state = next_state
            episode_reward += reward
            running_reward = running_reward * 0.99 + episode_reward * 0.01
            # rewards.append(episode_reward)
            if frame_idx % 500 == 0:
                plot(frame_idx, episode_rewards)

        print('Episode: ', i_episode, '| Episode Reward: ', episode_reward, '| Running Reward: ', running_reward)
        episode_rewards.append(episode_reward)
        if UPDATE == 'Approach0':
            Update0(rewards)
        if UPDATE == 'Approach1':
            Update1(rewards)
