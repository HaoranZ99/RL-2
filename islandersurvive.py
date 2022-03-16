import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import gym
import custom_gym

import random, datetime
from pathlib import Path

from logger import Logger

# Hyper Parameters

env = gym.make('Island-v0')
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

SAVE_DIR = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
SAVE_DIR.mkdir(parents=True)

EPISODES = 2000000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.batch_size = 32
        self.lr = 0.00025
        self.epsilon = 1
        self.epsilon_decay = 0.99999975
        self.epsilon_min = 0.1
        self.gamma = 0.9
        self.target_replace_iter = 1e4
        self.memory_capacity = 100000
        self.save_every = 5e6


        # 建立 target net 和 eval net 还有 memory
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.memory_capacity, N_STATES * 2 + 2))     # initialize memory

        self.net = Net().float()
        # 选择优化方法和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() > self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 如果目前步数小，直接跳过
        if self.memory_counter <= self.memory_capacity:
            return None, None
        
        if self.learn_step_counter % self.save_every == 0:
            self.save()
        
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return q_eval.mean().item(), loss.item()
    
    # 保存模型
    def save(self):
        save_path = SAVE_DIR / f"island_net_{int(self.learn_step_counter // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.epsilon
            ),
            save_path
        )
        print(f"IslandNet saved to {save_path} at step {self.learn_step_counter}")

dqn = DQN()

print('\nCollecting experience...')

# sum_r = 0
logger = Logger(SAVE_DIR)

for i_episode in range(EPISODES):

    s = env.reset()

    while True:

        a = dqn.choose_action(s)

        # 选动作, 得到环境反馈
        s_, r, done, _ = env.step(a)

        # 存记忆
        dqn.store_transition(s, a, r, s_)

        q, loss = dqn.learn()

        logger.log_step(r, loss, q)

        if done:
            break
        s = s_ # 将下一个state的值传到下一次循环
    
    logger.log_episode()

    if i_episode % 20 == 0:
        logger.record(
            episode=i_episode,
            epsilon=dqn.epsilon,
            step=dqn.learn_step_counter
        )