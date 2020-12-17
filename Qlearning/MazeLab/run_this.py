# -*- encoding: utf-8 -*-
"""
@File    : run_this.py
@Time    : 2020/11/29 
@Author  : Yuan Yifu
"""
from maze_env import Maze
from q_learning_brain import Qlearning


def run():
    for episode in range(200):  # repeat for each episode in episodes
        # initialize: S ← first state of episode
        observation = env.reset()

        # repeat for every step of episode, until S is terminal state
        done = False
        while not done:
            # fresh env
            env.render()

            # choose action(epsilon greedy)
            action = RL.choose_action(str(observation))

            # R, S' = perform_action(S, A)
            observation_, r, done = env.step(action)

            # update Q table -> Q learning
            RL.learn(str(observation), action, r, str(observation_))

            # S ← S'
            observation = observation_

    print("game over")
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = Qlearning(action_space=list(range(env.n_actions)))
    env.after(200, run)
    env.mainloop()
