import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np

class ReplayBuffer:
    def __init__(self,capacity) -> None:
        self.capacity=capacity # 经验回放的容量
        self.buffer=[]
        self.position=0
    
    def push(self,state,action,reward,next_state,done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer)<self.capacity:
            self.buffer.append(None)
        self.buffer[self.position]=(state, action, reward, next_state, done)
        self.position=(self.position+1) % self.capacity
    
    def sample(self,batch_size):
        batch=random.sample(self.buffer,batch_size)
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
class DQN:
    def __init__(self,state_dim,action_dim,cfg) -> None:
        self.state_dim=state_dim
        self.action_dim=action_dim
        # self.action_dim=action_dim
        self.device=cfg.device
        self.gamma=cfg.gamma    # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size=cfg.batch_size
        # Network
        self.policy_net=model.MLP(state_dim,action_dim).to(self.device)
        self.target_net=model.MLP(state_dim,action_dim).to(self.device)
        # copy
        for target_param,param in zip(self.target_net.parameters(),self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=cfg.lr)
        self.memory=ReplayBuffer(cfg.memory_capacity) # 经验回放
    
    def choose_action(self,state):
        self.frame_idx+=1
        if random.random()>self.epsilon(self.frame_idx):
            with torch.no_grad():
                state=torch.tensor([state],device=self.device,dtype=torch.float32)
                q_values=self.policy_net(state)
                action=q_values.max(1)[1].item()    # 选择Q值最大的动作
        else:
            action=random.randrange(self.action_dim)
        return action
    
    def update(self):
        # 当memory中不满足一个批量时，不更新策略
        if len(self.memory)<self.batch_size:
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch,action_batch,reward_batch,next_state_batch,done_batch=self.memory.sample(self.batch_size)
        # 转为张量
        state_batch=torch.tensor(state_batch,device=self.device,dtype=torch.float)
        action_batch=torch.tensor(action_batch,device=self.device).unsqueeze(1)
        reward_batch=torch.tensor(reward_batch,device=self.device,dtype=torch.float)
        next_state_batch=torch.tensor(next_state_batch,device=self.device,dtype=torch.float)
        done_batch=torch.tensor(np.float32(done_batch), device=self.device)
        
        q_values=self.policy_net(state_batch).gather(dim=1,index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values=self.target_net(next_state_batch).max(1)[0].detach() # 计算下一时刻的状态(s_t_,a)对应的Q值
                       
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expect_q_values=reward_batch+self.gamma*next_q_values*(1-done_batch)
        loss=nn.MSELoss()(q_values,expect_q_values.unsqueeze(1))        
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
    
    def save(self,path):
        torch.save(self.target_net.state_dict(),path+'dqn_checkpoint.pth')
        
    def load(self,path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param,param in zip(self.target_net.parameters(),self.policy_net.parameters()):
            param.data.copy_(target_param.data)    