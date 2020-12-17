# -*- encoding: utf-8 -*-
"""
@File    : run_this.py
@Time    : 2020/12/4 
@Author  : Yuan Yifu
"""
import gym
from DQN_brain import DQN
env = gym.make('CartPole-v0')
env = env.unwrapped
# print(env.action_space.n) 2
# print(env.observation_space.shape[0]) 4
dqn = DQN()
MEMORY_CAPACITY = 2000


def run():
    print('\nCollecting experience...\n')
    for i_episode in range(400):  # repeat for each episode in episodes
        s = env.reset()  # get start s of start state S
        ep_r = 0
        while True:  # repeat for each step of episode
            env.render()
            a = dqn.choose_action(s)  # choose action

            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            # store transition in buffer
            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                # until S is terminal state
                break

            # get next state
            s = s_


if __name__ == '__main__':
    run()
