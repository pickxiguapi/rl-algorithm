# -*- encoding: utf-8 -*-
"""
@File    : te_DDPG_model.py
@Time    : 2020/12/14 
@Author  : Yuan Yifu
"""
# from puckworld_continuous import PuckWorldEnv
from puckworld_con_enemy import PuckWorldEnv
from DDPG_agent import DDPGAgent
from utils import learning_curve

env = PuckWorldEnv()
agent = DDPGAgent(env)

agent.load_models(200)

data = agent.running(episode_num=100, display=False, explore=True)

learning_curve(data)