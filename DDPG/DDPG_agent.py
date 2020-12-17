# -*- encoding: utf-8 -*-
"""
@File    : DDPG_agent.py
@Time    : 2020/12/12 
@Author  : Yuan Yifu
"""
from gym import Env
from utils import OrnsteinUhlenbeckActionNoise, hard_update, soft_update
from DDPG_net import Actor, Critic
import torch
import numpy as np
import torch.nn.functional as F
import random
from collections import defaultdict
from tqdm import tqdm


class Transition(object):
    def __init__(self, s=None, a=None, reward=None, done=None, s_=None):
        self.data = [s, a, reward, done, s_]

    def __str__(self):
        return "s:{0:<3} a:{1:<3} r:{2:<4} done:{3:<5} s_:{4:<3}". \
                format(str(self.data[0]), str(self.data[1]), str(self.data[2]), str(self.data[3]), str(self.data[4]))

    @property
    def s(self): return self.data[0]

    @property
    def a(self): return self.data[1]

    @property
    def reward(self): return self.data[2]

    @property
    def done(self): return self.data[3]

    @property
    def s_(self): return self.data[4]


class Agent(object):
    """
        Agent 基类
    """
    def __init__(self, env: Env = None,
                 capacity=1000):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env  # 建立对环境对象的引用
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None

        self.state = None  # 个体的当前状态

        # experience replay
        self.max_capacity = capacity
        self.pointer = 0
        self.memory = defaultdict(Transition)

    def store_transition(self, trans):
        index = self.pointer % self.max_capacity  # replace the old memory with new memory
        index = int(index)
        self.memory[index] = trans
        self.pointer += 1

    def step(self, a):
        s = self.state
        s_, r, done, info = self.env.step(a)

        trans = Transition(s, a, r, done, s_)
        self.store_transition(trans)
        self.state = s_
        return s_, r, done, info

    def sample(self, batch_size=64):
        """
        随机取样batch_size个样本，保证经验池满了之后再开始取样
        :param batch_size:
        :return:
        """
        sample_trans = []
        for _ in range(batch_size):
            index = random.randint(0, self.max_capacity - 1)
            sample_trans.append(self.memory[index])
        return sample_trans

    @property
    def total_trans(self):
        return len(self.memory)


# mu: Actor mu': Target Actor
# Q: Critic Q': Target Critic
class DDPGAgent(Agent):
    def __init__(self, env: Env,
                 capacity=10000,
                 batch_size=128,
                 action_limit=1,
                 learning_rate=0.001,
                 gamma=0.99,):
        super(DDPGAgent, self).__init__(env, capacity)
        self.state_dim = env.observation_space.shape[0]  # 状态特征数
        self.action_dim = env.action_space.shape[0]  # 动作特征数
        self.action_lim = action_limit  # 动作值限制
        self.batch_size = batch_size  # batch num
        self.learning_rate = learning_rate  # 学习率
        self.gamma = gamma
        self.tau = 0.001  # soft update
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_lim)
        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print("Models saved successfully")

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print("Models loaded successfully")

    def _learn_from_memory(self):
        """
        learn from memory
        :return:
        """

        # 获取batch size个transition
        trans_pieces = self.sample(self.batch_size)
        s = np.vstack([x.s for x in trans_pieces])
        a = np.array([x.a for x in trans_pieces])
        r = np.array([x.reward for x in trans_pieces])
        s_ = np.vstack([x.s_ for x in trans_pieces])

        # Set yi = r + gamma * Q'(s_, mu'(s_))
        a_ = self.target_actor.forward(s_).detach()
        Q_ = torch.squeeze(self.target_critic.forward(s_, a_).detach())
        r = torch.from_numpy(r)
        y_target = r + self.gamma * Q_
        y_target = y_target.type(torch.FloatTensor)

        # y_pred = Q(s, a)
        a = torch.from_numpy(a)
        y_pred = torch.squeeze(self.critic.forward(s, a))

        # Update critic by minimizing the loss: L = smooth_l1_loss(y_pred, y_target)
        loss_critic = F.smooth_l1_loss(y_pred, y_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Update the actor policy using the sampled policy gradient
        # grad J = sum(grad Q(s, mu(s)))
        """
            $$
                \left.\left.\nabla_{\theta^{\mu}} J \approx \frac{1}{N} \sum_{i} \nabla_{a} Q\left(s, a \mid \theta^{Q}\right)\right|_{s=s_{i}, a=\mu\left(s_{i}\right)} \nabla_{\theta^{\mu}} \mu\left(s \mid \theta^{\mu}\right)\right|_{s_{i}}
            $$
        """
        pred_a = self.actor.forward(s)  # mu(s)
        # 梯度上升
        loss_actor = -1 * torch.sum(self.critic.forward(s, pred_a))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

        return loss_critic.item(), loss_actor.item()

    def running(self, episode_num=800, display=False, explore=True):
        step_list = []
        episode_reward_list = []

        for i_episode in range(episode_num):
            self.state = np.float64(self.env.reset())
            done = False

            # init episode loss (loss: loss_total / step_num)
            loss_critic, loss_actor = 0, 0
            # init step_num
            step_num = 0
            # init episode reward
            episode_reward = 0
            # episode index
            episode_index = i_episode + 1

            while not done:
                # get state
                s = self.state

                # Select action
                if explore:
                    # a = mu(s) + N
                    a = self.get_exploration_action(s)
                else:
                    # a = mu(s)
                    a = self.get_action(s)

                # Execute action a and observe reward r and observe new state s_
                # Store transition(s, a, r, s_) in experience replay
                s_, r, done, info = self.step(a)

                # update reward
                episode_reward += r

                if display:
                    self.env.render()

                # calculate loss
                if self.total_trans >= self.max_capacity:

                    loss_c, loss_a = self._learn_from_memory()
                    loss_critic += loss_c
                    loss_actor += loss_a

                # step num ++
                step_num += 1
                if step_num >= 4000:
                    done = True

            # an episode end, calculate some info
            loss_critic /= step_num
            loss_actor /= step_num
            step_list.append(step_num)
            episode_reward_list.append(episode_reward)

            print("episode:{:3}, step num:{}, reward:{}, loss critic:{:4.3f}, loss_actor:{:4.3f}".
                  format(episode_index, step_num, episode_reward, loss_critic, loss_actor))

            if explore and episode_index % 100 == 0:
                self.save_models(episode_index)

        return step_list, episode_reward_list

    def get_exploration_action(self, state):
        """
        exploration 探索
        :param state: state numpy float64
        :return:
        """
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        new_action = new_action.clip(min=-1 * self.action_lim,
                                     max=self.action_lim)
        return new_action

    def get_action(self, state):
        return self.actor.forward(state).detach().data.numpy()






