# -*- encoding: utf-8 -*-
"""
@File    : run_cartpole.py
@Time    : 2020/12/5 
@Author  : Yuan Yifu
"""
import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(2)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

"""
print(env.action_space)  # 2
print(env.observation_space)  # Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
print(env.observation_space.high)
print(env.observation_space.low)
"""

RL = PolicyGradient()
loss_list = list()

for i_episode in range(3000):

    observation = env.reset()
    while True:
        if RENDER:
            # env.render()
            pass

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.rewards)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            loss, vt = RL.learn()
            break

        observation = observation_
