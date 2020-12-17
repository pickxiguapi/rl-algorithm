# -*- encoding: utf-8 -*-
"""
@File    : run_this.py
@Time    : 2020/11/29 
@Author  : Yuan Yifu
"""
from maze_env import Maze
from sarsa_brain import Sarsa


def run():
    for episode in range(200):  # repeat for each episode in episodes
        # initialize: S ← first state of episode
        observation = env.reset()
        # choose action(epsilon greedy)
        action = RL.choose_action(str(observation))

        # repeat for every step of episode, until S is terminal state
        done = False
        while not done:
            # fresh env
            env.render()

            # R, S' = perform_action(S, A)
            observation_, r, done = env.step(action)

            # A' = policy(Q, S')
            action_ = RL.choose_action(str(observation_))

            # update Q table -> Sarsa
            RL.learn(str(observation), action, r, str(observation_), action_)

            # S ← S'; A ← A'
            observation = observation_
            action = action_

    print("game over")
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = Sarsa(action_space=list(range(env.n_actions)))
    env.after(200, run)
    env.mainloop()
