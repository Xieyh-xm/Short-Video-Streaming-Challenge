import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, action_dim, cfg) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.action_dim=action_dim
        self.device = cfg.device
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        # Network
        self.policy_net = CNN(state_dim, action_dim).to(self.device)
        self.target_net = CNN(state_dim, action_dim).to(self.device)
        # copy
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)  # 经验回放

    def choose_action(self, state):
        self.frame_idx += 1
        chunk_last = state[0, 225:230]
        mask = np.zeros(15)
        for i in range(5):
            for j in range(3):
                if chunk_last[i] != 0.0:
                    mask[i * 3 + j] = 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                # state = torch.tensor([state], device=self.device, dtype=torch.float32)
                # state = torch.tensor(state, device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                q_values[0][mask == 0] = -float("inf")
                action = q_values.max(1)[1].item()  # 选择Q值最大的动作
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        # 当memory中不满足一个批量时，不更新策略
        if len(self.memory) < self.batch_size:
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        # 转为张量
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float).squeeze(1)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float).squeeze(1)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # 计算当前状态(s_t,a)对应的Q(s_t, a)
        # next_q_values = self.target_net(next_state_batch).max(1)[0].detach()  # 计算下一时刻的状态(s_t_,a)对应的Q值
        next_q_values = self.target_net(next_state_batch)
        chunk_last = state_batch[:].numpy()
        chunk_last = chunk_last[:, 225:230]
        for i in range(len(state_batch)):
            mask = np.zeros(15)
            for j in range(5):
                for k in range(3):
                    if chunk_last[i][j] != 0.0:
                        mask[j * 3 + k] = 1
            next_q_values[i][mask == 0] = -999999
        next_q_values = next_q_values.max(1)[0].detach()
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expect_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expect_q_values.unsqueeze(1))
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        ''' 初始化q网络，为全连接网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        '''
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        ''' 初始化q网络，为CNN网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        '''
        super(CNN, self).__init__()
        # Conv1d
        self.layer1_shape_1 = 4  # conv1d的输出通道个数
        self.layer1_shape_2 = 8
        self.numFcInput = (2 * 8) * self.layer1_shape_1 + (2 * 8) * self.layer1_shape_2 + 15
        self.layer2_shape = 128
        # 过去10个chunk的吞吐量throughput 1x10
        self.tConv1d = nn.Conv1d(1, self.layer1_shape_1, 3)  # 输入通道=1
        # 过去10个chunk的下载时刻playtime 1x10
        self.pConv1d = nn.Conv1d(1, self.layer1_shape_1, 3)
        # 5个视频未来10个chunk的3级video_size 15x10
        # self.vConv1d = nn.Conv1d(15, self.layer1_shape_2, 3)
        self.vConv1d = nn.Conv1d(5, self.layer1_shape_2, 3)
        # 5个视频未来10个chunk的conditional_retent_rate 5x10
        self.rConv1d = nn.Conv1d(5, self.layer1_shape_2, 3)

        # 5个视频的buffer 5x1 (直接输入全连接)
        # 5个视频剩余的chunk数remain chunks 5x1 (直接输入全连接)
        # 5个视频上一个chunk的质量等级last_level 5x1 (直接输入全连接)

        # 2层全连接
        self.fc1 = nn.Linear(self.numFcInput, self.layer2_shape)
        self.p_output = nn.Linear(self.layer2_shape, action_dim)

    def forward(self, inputs):
        throughput = inputs[:, 0:10].unsqueeze(1)
        playtime = inputs[:, 10:20].unsqueeze(1)
        video_size = torch.reshape(inputs[:, 20:170], (inputs[:, 20:170].shape[0], 15, 10))
        video_size = video_size[:, [0, 3, 6, 9, 12]]
        ret_rate = torch.reshape(inputs[:, 170:220], (inputs[:, 170:220].shape[0], 5, 10))

        # 过去10个chunk的吞吐量throughput 1x10
        # throughputConv = F.relu(self.tConv1d(inputs[:, 0:1, :]), inplace=True)
        throughputConv = F.relu(self.tConv1d(throughput), inplace=True)
        # 过去10个chunk的下载时刻playtime 1x10
        # playtimeConv = F.relu(self.pConv1d(inputs[:, 1:2, :]), inplace=True)
        playtimeConv = F.relu(self.pConv1d(playtime), inplace=True)
        # 5个视频未来10个chunk的3级video_size 15x10
        # video_sizeConv = F.relu(self.vConv1d(inputs[:, 2:17, :]), inplace=True)
        video_sizeConv = F.relu(self.vConv1d(video_size), inplace=True)
        # 5个视频未来10个chunk的conditional_retent_rate 5x10
        # ret_rateConv = F.relu(self.rConv1d(inputs[:, 17:22, :]), inplace=True)
        ret_rateConv = F.relu(self.rConv1d(ret_rate), inplace=True)

        # flatten
        throughput_flatten = throughputConv.view(throughputConv.shape[0], -1)
        playtime_flatten = playtimeConv.view(playtimeConv.shape[0], -1)
        video_size_flatten = video_sizeConv.view(video_sizeConv.shape[0], -1)
        ret_rate_flatten = ret_rateConv.view(ret_rateConv.shape[0], -1)
        # 5个视频的buffer 5x1
        # 5个视频剩余的chunk数remaining chunks 5x1
        # 5个视频上一chunk的质量等级last_level 5x1
        # buffer_flatten = inputs[:, 220:225].view(inputs[:, 220:225].shape[0], -1)
        # remain_chunks_flatten = inputs[:, 225:230].view(inputs[:, 225:230, :].shape[0], -1)
        # last_level_flatten = inputs[:, 230:235].view(inputs[:, 230:235, :].shape[0], -1)
        buffer_flatten = inputs[:, 220:225] / 1000.0
        remain_chunks_flatten = inputs[:, 225:230] / 10.0
        last_level_flatten = inputs[:, 230:235]

        merge = torch.cat([throughput_flatten, playtime_flatten, video_size_flatten, ret_rate_flatten, buffer_flatten,
                           remain_chunks_flatten, last_level_flatten], 1)
        fc1Out = F.relu(self.fc1(merge), inplace=True)
        q_value = self.p_output(fc1Out)
        return q_value
