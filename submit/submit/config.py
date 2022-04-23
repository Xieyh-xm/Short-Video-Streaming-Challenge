import sys
import torch
import datetime
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class Config:
    '''超参数
    '''
    def __init__(self) -> None:
        ################################## 环境超参数 ###################################
        self.algo_name='DQN'
        self.device=torch.device("cuda" if torch.cuda.is_available() else "CPU")
        self.seed=10
        self.train_eps=200
        self.test_eps=30
        ################################################################################
        
        ################################## 算法超参数 ###################################
        self.gamma=0.95
        self.eplison_start=0.90
        self.eplison_end=0.01
        self.eplison_decay=500
        self.lr=0.0001
        self.memory_capacity=100000
        self.batch_size=64
        self.target_update=4
        ################################################################################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        ################################################################################
        
            