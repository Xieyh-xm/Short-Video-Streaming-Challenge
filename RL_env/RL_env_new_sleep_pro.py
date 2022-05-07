import sys
import itertools
import numpy

sys.path.append('../ACM_MM2022/simulator/')
import os
import random
import numpy as np
from numpy import *
import glob
from queue import Queue
import torch
import math
import json
from simulator import controller as env, short_video_load_trace
from gym import spaces

# from myPrint import print_debug

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
SUMMARY_DIR = '../ACM_MM2022/logs'
LOG_FILE = '../ACM_MM2022/logs/log.txt'

# QoE arguments
alpha = 1
beta = 1.85
gamma = 1
theta = 0.5
ALL_VIDEO_NUM = 7
STATE_DIMENSION = 1
HISTORY_LENGTH = 235
PRINT_FLAG = False
TAU = 50


class RLEnv:
    def __init__(self):
        self.state = torch.zeros(STATE_DIMENSION, HISTORY_LENGTH)
        self.all_cooked_time = []
        self.all_cooked_bw = []
        net_trace_dict = './data/network_traces/mix_medium/'
        video_trace_dict = './data/short_video_size'
        ret_trace_dict = './data/user_ret'
        self.user_sample_id = 0
        self.trace_id = 0
        self.all_cooked_time, self.all_cooked_bw = short_video_load_trace.load_trace(net_trace_dict)
        self.seeds = np.random.randint(10000, size=(7, 2))
        self.net_env = env.Environment(self.user_sample_id, self.all_cooked_time[self.trace_id],
                                       self.all_cooked_bw[self.trace_id], ALL_VIDEO_NUM,
                                       self.seeds)
        # self.action_space = spaces.Discrete(16)
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Discrete(30)

        # ======== extra param ========
        self.last_bitrate = -1
        # caculate reward
        self.last_play_video_id = 0
        self.last_play_chunk_idx = -1
        self.last_residual_time = 0
        self.play_chunk_list = [[] for i in range(5)]  # 读入当前player已经下载的chunk
        # state 状态数组
        self.past_10_throughput = []
        self.past_10_delay = []
        self.future_videosize = []  # 五个视频三个质量未来10个chunk videosize 形式为[5][3][10]
        self.conditional_retent_rate = []  # 条件转移概率 [5][10]
        self.cache = []  # buffer  [5]
        self.left_chunk_cnt = []  # 剩余chunk数 [5]
        self.last_5_bitrate = []  # 五个视频上个质量等级 [5]
        self.offset = 0  # 视频偏移量

    def reset(self, trace_id, user_sample_id):
        self.seeds = np.random.randint(user_sample_id + 1, size=(7, 2))
        self.net_env = env.Environment(user_sample_id + 1, self.all_cooked_time[trace_id],
                                       self.all_cooked_bw[trace_id], ALL_VIDEO_NUM,
                                       self.seeds)
        # state 状态数组
        self.left_chunk_cnt = []  # 剩余chunk数
        # self.state = torch.zeros((1, STATE_DIMENSION, HISTORY_LENGTH))
        self.last_bitrate = -1
        self.last_play_video_id = 0
        self.last_play_chunk_idx = -1
        self.last_residual_time = 0
        # 填充state
        self.past_10_throughput = [1000000 for i in range(10)]
        self.past_10_delay = [1000 for i in range(10)]
        self.cache = [0 for i in range(5)]
        self.last_5_bitrate = [0 for i in range(5)]
        self.future_videosize = [0 for i in range(5)]
        self.conditional_retent_rate = [0 for i in range(5)]
        self.left_chunk_cnt = [0 for i in range(5)]
        self.offset = 0
        self.state = torch.zeros(1, 30)

        # 获取五个视频三个质量未来10个chunk videosize
        for i, player in enumerate(self.net_env.players):
            self.future_videosize[i] = player.video_size[0][0]
            self.conditional_retent_rate[i] = float(player.user_retent_rate[i])
            self.left_chunk_cnt[i] = player.chunk_num

        mergelist = self.past_10_throughput[
                    5:10] + self.future_videosize + self.conditional_retent_rate + self.cache + self.left_chunk_cnt + self.last_5_bitrate
        mergelist = torch.Tensor(mergelist)
        self.state[0, 0:5] = mergelist[0:5] / 1000000
        self.state[0, 5:10] = mergelist[5:10] / 1000000
        self.state[0, 10:15] = mergelist[10:15]
        self.state[0, 15:20] = mergelist[15:20] / 1000
        self.state[0, 20:25] = mergelist[20:25] / 10
        self.state[0, 25:30] = mergelist[25:30]
        return self.state

    def step(self, action):
        switch_flag = False
        sleep_flag = False
        quality = 0
        smooth = 0
        bit_rate = int(action[1])
        download_video_id = int(action[0])
        download_video_id += self.offset
        sleep_time = 0
        if action[2]:  # sleep time flag
            sleep_time = TAU
        done = False

        if sleep_time > 0:
            sleep_flag = True
        else:
            sleep_flag = False
            # the last chunk id that user watched
            max_watch_chunk_id = self.net_env.user_models[
                download_video_id - self.net_env.get_start_video_id()].get_watch_chunk_cnt()
            # last downloaded chunk id
            download_chunk = self.net_env.players[
                download_video_id - self.net_env.get_start_video_id()].get_chunk_counter()
            if max_watch_chunk_id >= download_chunk:  # the downloaded chunk will be played
                quality = VIDEO_BIT_RATE[bit_rate]
                # 保存下载的chunk quality
                self.play_chunk_list[download_video_id - self.last_play_video_id].append(quality)
                if self.last_bitrate != -1:  # is not the first chunk to play
                    smooth = abs(quality - VIDEO_BIT_RATE[self.last_bitrate])
                self.last_bitrate = bit_rate

        # 和环境交互
        delay, rebuf, video_size, end_of_video, \
        play_video_id, waste_bytes = self.net_env.buffer_management(download_video_id, bit_rate, sleep_time)
        # 获取state状态
        ''' 1. 更新过去的吞吐量 '''
        ''' 2. 更新过去的delay '''
        if not sleep_flag:
            self.past_10_throughput.pop(0)
            self.past_10_throughput.append(video_size / delay * 1000.0)
            self.past_10_delay.pop(0)
            self.past_10_delay.append(delay)
        # 如果发生视频跳转 需要更新新播放视频和新下载视频的数据
        # play_video_id,download_id = 0-6 此时视频id已经+1 ，
        player_length = len(self.net_env.players)

        ''' 4. 更新缓冲区'''
        for i in range(5):
            for i in range(5):
                if i >= player_length:
                    self.cache[i] = 0
                else:
                    self.cache[i] = self.net_env.players[i].buffer_size

        ''' 5. 更新剩余chunk '''
        for i in range(5):
            if i >= player_length:
                self.left_chunk_cnt[i] = 0
            else:
                self.left_chunk_cnt[i] = self.net_env.players[i].get_remain_video_num()

        ''' 3. 更新future_videosize '''
        for i in range(5):
            if i >= player_length:
                self.future_videosize[i] = 0
            else:
                if self.left_chunk_cnt[i] > 0:
                    chunk_size = self.net_env.players[i].get_future_video_size(1)
                    self.future_videosize[i] = chunk_size[0][0]
                else:
                    self.future_videosize[i] = 0

        ''' 6. 更新条件概率 '''
        for i in range(5):
            if i >= player_length:
                self.conditional_retent_rate[i] = 0
            else:
                playing_chunk = math.ceil(self.net_env.players[i].get_play_chunk())
                retent_rate_of_download = float(self.net_env.players[i].user_retent_rate[-self.left_chunk_cnt[i]])
                retent_rate_of_play = float(self.net_env.players[i].user_retent_rate[playing_chunk])
                self.conditional_retent_rate[i] = retent_rate_of_download / retent_rate_of_play

        ''' 7. 更新上一个视频的码率等级'''
        # 发生跳转
        if play_video_id > self.last_play_video_id and play_video_id < ALL_VIDEO_NUM:
            jump_number = play_video_id - self.last_play_video_id  # 一次跳转的视频个数
            self.offset =play_video_id
            # 7.1 更新上个质量等级
            for i in range(jump_number):
                self.last_5_bitrate.pop(0)
                self.last_5_bitrate.append(0)
            if download_video_id - self.offset >= 0 and not sleep_flag:
                self.last_5_bitrate[download_video_id - self.offset] = bit_rate
            self.last_play_video_id = play_video_id
        # 未发生跳转
        if play_video_id == self.last_play_video_id:
            # 7.2 更新上个质量等级
            if not sleep_flag:
                self.last_5_bitrate[download_video_id - self.offset] = bit_rate

        one_step_QOE = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000 - theta * 8 * video_size / 1000000.

        if play_video_id >= ALL_VIDEO_NUM:  # 全部播放完为done
            done = True

        mergelist = self.past_10_throughput[
                    5:10] + self.future_videosize + self.conditional_retent_rate + self.cache + self.left_chunk_cnt + self.last_5_bitrate
        mergelist = torch.Tensor(mergelist)

        self.state[0, 0:5] = mergelist[0:5] / 1000000
        self.state[0, 5:10] = mergelist[5:10] / 1000000
        self.state[0, 10:15] = mergelist[10:15]
        self.state[0, 15:20] = mergelist[15:20] / 1000
        self.state[0, 20:25] = mergelist[20:25] / 10
        self.state[0, 25:30] = mergelist[25:30]

        return self.state, one_step_QOE, done
