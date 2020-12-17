# -*- encoding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2020/12/12 
@Author  : Yuan Yifu
"""
import numpy as np
import matplotlib.pyplot as plt


def soft_update(target_net, source_net, tau):
    """
    更新副本的参数时不再是直接复制，而是采用一种缓慢更新的方式
    tau << 1
    target = tau * source + (1 - tau) * target
    :param target_net: 目标网络
    :param source_net: 源网络
    :param tau: 更新速率 tau<<1
    :return: none
    """

    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            tau * source_param + (1 - tau) * target_param
        )


def hard_update(target_net, source_net):
    """
    完全复制更新参数
    :param target_net:
    :param source_net:
    :return:
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param)


class OrnsteinUhlenbeckActionNoise(object):
    """
        Ornstein-Uhlenbeck 噪声函数
    """
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


def learning_curve(data, title="", x_name="", y_name="", y_legend=""):

    fig, ax = plt.subplots()
    reward = data[1]
    x = [i for i in range(len(reward))]
    ax.plot(x, reward, label=y_legend)

    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=1)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend()

    plt.show()
