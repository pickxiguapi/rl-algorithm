# -*- encoding: utf-8 -*-
"""
@File    : run_cartpole.py
@Time    : 2020/12/6 
@Author  : Yuan Yifu
"""
import gym
from A2C_brain import A2C
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env = env.unwrapped

"""
print(env.action_space)  # 2
print(env.observation_space)  # Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
print(env.observation_space.high)
print(env.observation_space.low)
"""

RL = A2C()
loss_list = list()
running_reward = 10
for i_episode in range(3000):

    # init

    observation = env.reset()
    ep_r = 0
    while True:
        if RENDER:
            # env.render()
            pass

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        ep_r += reward
        RL.add_rewards(reward)
        if done:
            break

        observation = observation_

    # update cumulative reward
    running_reward = 0.05 * ep_r + (1 - 0.05) * running_reward

    # perform back propagation
    RL.learn()

    # log results
    if i_episode % 1 == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            i_episode, ep_r, running_reward))

