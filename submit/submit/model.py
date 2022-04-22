import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np

class MLP(nn.module):
    def __init__(self,state_dim,action_dim):
        ''' 初始化q网络，为全连接网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        '''
        super(MLP,self).__init__()
        self.fc1=nn.Linear(state_dim,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,action_dim)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)

