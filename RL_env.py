import sys

sys.path.append('./simulator/')
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

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
SUMMARY_DIR = 'logs'
LOG_FILE = 'logs/log.txt'

# QoE arguments
alpha = 1
beta = 1.85
gamma = 1
theta = 0.5
ALL_VIDEO_NUM = 7


class RLEnv:
    def __init__(self):
        self.all_cooked_time = []
        self.all_cooked_bw = []
        net_trace_dict = './data/network_traces/medium'
        video_trace_dict = './data/short_video_size'
        ret_trace_dict = './data/user_ret'
        self.user_sample_id = 0
        self.trace_id = 0
        self.all_cooked_time, self.all_cooked_bw = short_video_load_trace.load_trace(net_trace_dict)
        self.seeds = np.random.randint(10000, size=(7, 2))
        self.net_env = env.Environment(self.user_sample_id, self.all_cooked_time[self.trace_id],
                                       self.all_cooked_bw[self.trace_id], ALL_VIDEO_NUM,
                                       self.seeds)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=np.zeros((STATE_DIMENSION, HISTORY_LENGTH)),
            high=np.ones((STATE_DIMENSION, HISTORY_LENGTH)),
            dtype=np.float64)
        self.state = torch.zeros((1, STATE_DIMENSION, HISTORY_LENGTH))
        # ======== extra param ========
        self.last_bitrate = -1
        # caculate reward
        self.last_play_video_id = 0
        self.last_play_chunk_idx = 0
        self.last_residual_time = 0
        self.play_chunk_list = [[] for i in range(5)]  # 读入当前player已经下载的chunk

    def reset(self, trace_id, user_sample_id):
        self.net_env = env.Environment(user_sample_id, self.all_cooked_time[trace_id],
                                       self.all_cooked_bw[trace_id], ALL_VIDEO_NUM,
                                       self.seeds)
        self.state = torch.zeros((1, STATE_DIMENSION, HISTORY_LENGTH))
        self.last_bitrate = -1
        self.last_play_video_id = 0
        self.last_play_chunk_idx = 0
        self.last_residual_time = 0

    def step(self, action):
        quality = 0
        smooth = 0
        last_played_chunk = -1  # record the last played chunk
        bit_rate =
        download_video_id =
        sleep_time =
        done = False

        if sleep_time == 0:
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
                    # print("downloading ", download_video_id, "chunk ", download_chunk, ", bitrate switching from ", last_bitrate, " to ", bit_rate)
                self.last_bitrate = bit_rate

        # 和环境交互
        delay, rebuf, video_size, end_of_video, \
        play_video_id, waste_bytes = self.net_env.buffer_management(download_video_id, bit_rate, sleep_time)
        # print(delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes)
        # print log info of the last operation
        if play_video_id < ALL_VIDEO_NUM:
            # the operation results
            current_chunk = self.net_env.players[0].get_play_chunk()
            # print(current_chunk)
            current_bitrate = self.net_env.players[0].get_video_quality(max(int(current_chunk - 1e-10), 0))

        # =================== reward计算 =====================
        # 1. 计算已播放的总数据量
        sum_of_smooth = self.play_chunk_list[0][
                            self.last_play_chunk_idx + 1] * self.last_residual_time  # 上次播放了一部分的chunk
        sum_of_quality = 0
        num_of_video = min(play_video_id, ALL_VIDEO_NUM) - self.last_play_video_id  # 播放的视频个数
        for i in range(num_of_video):
            # 首个视频上一步已经播放一部分chunk
            if i == 0:
                sum_of_quality += sum(self.play_chunk_list[i][self.last_play_chunk_idx + 1:])
                num_of_chunk_per_video = len(self.play_chunk_list[i][self.last_play_chunk_idx:])
                for j in range(num_of_chunk_per_video):
                    sum_of_smooth += self.play_chunk_list[i][j] - self.play_chunk_list[i][j - 1]
            # 当前视频只播到current_chunk
            if i == num_of_video - 1:
                sum_of_quality += sum(self.play_chunk_list[i][:current_chunk])
                num_of_chunk_per_video = len(self.play_chunk_list[i][:current_chunk])
                for j in range(1, num_of_chunk_per_video):
                    sum_of_smooth += self.play_chunk_list[i][j] - self.play_chunk_list[i][j - 1]
            # 全部播放完
            sum_of_quality += sum(self.play_chunk_list[i])
            num_of_chunk_per_video = len(self.play_chunk_list[i][:])
            for j in range(1, num_of_chunk_per_video):
                sum_of_smooth += self.play_chunk_list[i][j] - self.play_chunk_list[i][j - 1]

        # 最后一个没完全放完的chunk播放了多少，取出小数部分
        delta_time = (self.net_env.players[0].play_timeline - self.last_residual_time) / 1000. - int(
            (self.net_env.players[0].play_timeline - self.last_residual_time) / 1000.)
        sum_of_quality += self.play_chunk_list[i][current_chunk] * delta_time
        self.last_residual_time = 1 - delta_time
        # 2. play_chunk_list更新，pop已经完整播过的视频，并补充[]
        for i in range(num_of_video - 1):
            self.play_chunk_list.pop(0)
            self.play_chunk_list.append([])

        # quality取平均码率（delay-ms）& smooth取平均
        quality_per_s = sum_of_quality / (delay / 1000.)
        smooth_per_s = sum_of_smooth / (delay / 1000.)

        reward = alpha * sum_of_quality / 1000. - beta * rebuf / 1000. - gamma * smooth_per_s / 1000.
        # reward = one_step_QoE - theta * 8 * video_size / 1000000
        self.last_play_video_id = min(play_video_id, ALL_VIDEO_NUM)
        self.last_play_chunk_idx = current_chunk

        if play_video_id >= ALL_VIDEO_NUM:
            done = True

        state =

        return state, reward, done
