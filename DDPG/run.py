# -*- encoding: utf-8 -*-
"""
@File    : run.py
@Time    : 2020/12/13 
@Author  : Yuan Yifu
"""

from puckworld_continuous import PuckWorldEnv
from DDPG_agent import DDPGAgent
from utils import learning_curve

env = PuckWorldEnv()
agent = DDPGAgent(env)

data = agent.running(episode_num=200, display=False, explore=True)

learning_curve(data)




