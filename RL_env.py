import sys
import itertools
import numpy

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
from myPrint import print_debug

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
SUMMARY_DIR = 'logs'
LOG_FILE = 'logs/log.txt'

# QoE arguments
alpha = 1
beta = 1.85
gamma = 1
theta = 0.5
ALL_VIDEO_NUM = 7
STATE_DIMENSION = 1
HISTORY_LENGTH = 235
PRINT_FLAG = False


class RLEnv:
    def __init__(self):
        self.state = torch.zeros(STATE_DIMENSION, HISTORY_LENGTH)
        self.all_cooked_time = []
        self.all_cooked_bw = []
        net_trace_dict = './data/network_traces/medium/'
        video_trace_dict = './data/short_video_size'
        ret_trace_dict = './data/user_ret'
        self.user_sample_id = 0
        self.trace_id = 0
        self.all_cooked_time, self.all_cooked_bw = short_video_load_trace.load_trace(net_trace_dict)
        self.seeds = np.random.randint(10000, size=(7, 2))
        self.net_env = env.Environment(self.user_sample_id, self.all_cooked_time[self.trace_id],
                                       self.all_cooked_bw[self.trace_id], ALL_VIDEO_NUM,
                                       self.seeds)
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Discrete(235)
        # self.observation_space = spaces.Box(
        #     low=np.zeros((STATE_DIMENSION, HISTORY_LENGTH)),
        #     high=np.ones((STATE_DIMENSION, HISTORY_LENGTH)),
        #     dtype=np.float64)

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
        # todo:reset
        self.net_env = env.Environment(user_sample_id, self.all_cooked_time[trace_id],
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
        self.past_10_throughput = [0 for i in range(10)]
        self.past_10_delay = [0 for i in range(10)]
        self.cache = [0 for i in range(5)]
        self.last_5_bitrate = [0 for i in range(5)]
        self.future_videosize = [[[0 for t in range(10)] for i in range(3)] for k in range(5)]

        self.conditional_retent_rate = [[0 for t in range(10)] for k in range(5)]
        self.left_chunk_cnt = [0 for i in range(5)]
        self.offset = 0
        self.state = torch.zeros(STATE_DIMENSION, HISTORY_LENGTH)
        # self.past_10_throughput = numpy.zeros(10)
        # self.past_10_delay = numpy.zeros(10)
        # self.cache = numpy.zeros(5)
        # self.last_bitrate = numpy.zeros(5)
        # self.future_videosize = numpy.zeros((5, 3, 10))
        # self.conditional_retent_rate = numpy.zeros((5, 10))
        # self.left_chunk_cnt = numpy.zeros(5)
        # 获取五个视频三个质量未来10个chunk videosize
        i = 0
        for player in self.net_env.players:
            # 获取五个视频三个质量未来10个chunk videosize
            length = min(10, len(player.video_size[0]))
            for j in range(length):
                self.future_videosize[i][0][j] = player.video_size[0][j]
                self.future_videosize[i][1][j] = player.video_size[1][j]
                self.future_videosize[i][2][j] = player.video_size[2][j]
                # 获取未来10个chunk 条件留存率
                self.conditional_retent_rate[i][j] = float(player.user_retent_rate[j])
            # 剩余chunk数
            self.left_chunk_cnt[i] = player.chunk_num
            i = i + 1

        mergelist1 = list(itertools.chain.from_iterable(self.future_videosize))
        mergelist1 = list(itertools.chain.from_iterable(mergelist1))
        mergelist2 = list(itertools.chain.from_iterable(self.conditional_retent_rate))
        mergelist = self.past_10_throughput + self.past_10_delay + mergelist1 + mergelist2 + self.cache + self.left_chunk_cnt + self.last_5_bitrate
        for i in range(235):
            self.state[0][i] = mergelist[i]

        return self.state

    def step(self, action):
        print_debug("============================================================================================")
        switch_flag = False
        sleep_flag = False
        quality = 0
        smooth = 0
        last_played_chunk = -1  # record the last played chunk
        player_length = 0

        download_video_id = action // 3
        bit_rate = action % 3
        sleep_time = 0

        flag = True
        # 避免无效动作
        if download_video_id >= len(self.net_env.players):
            flag = False
            download_video_id = len(self.net_env.players) - 1
        if self.net_env.players[download_video_id].get_remain_video_num() == 0:
            sleep_time = 500

        print_debug('修正前下载视频id = ' + str(download_video_id))
        download_video_id += self.offset
        print_debug('修正后下载视频id = ' + str(download_video_id))

        done = False
        jump_number = 0
        if sleep_time > 0:
            sleep_flag = True
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
        player_length = len(self.net_env.players)
        print_debug('此时视频序列实际长度为' + str(player_length))
        # 获取state状态
        # 1.过去的吞吐量
        # self.past_10_throughput = numpy.delete(self.past_10_throughput, 0)
        # self.past_10_throughput = numpy.append(self.past_10_throughput, video_size)
        self.past_10_throughput.pop(0)
        self.past_10_throughput.append(video_size / delay * 1000.0)
        # 2.过去的delay
        # self.past_10_delay = numpy.delete(self.past_10_delay, 0)
        # self.past_10_delay = numpy.append(self.past_10_delay, delay)
        self.past_10_delay.pop(0)
        self.past_10_delay.append(delay)
        # 如果发生视频跳转 需要更新新播放视频和新下载视频的数据
        # play_video_id,download_id = 0-6 此时视频id已经+1 ，

        if play_video_id > self.last_play_video_id and play_video_id < ALL_VIDEO_NUM:
            jump_number = play_video_id - self.last_play_video_id  # 一次跳转的视频个数
            self.offset += jump_number
            print_debug('当前视频播放id从' + str(self.last_play_video_id) + '跳转到' + str(play_video_id) + '跳转了' + str(
                play_video_id - self.last_play_video_id) + '个视频')
            print_debug('download_video_id=' + str(download_video_id) + 'offset=' + str(self.offset) + '剩余下载块' + str(
                self.net_env.players[
                    download_video_id - self.offset].get_remain_video_num()) + 'player_id = ' + str(play_video_id))
            # 检查player长度
            player_length = len(self.net_env.players)

            # 3.1 更新future_videosize
            if player_length == 5:  # 如果推荐列表为满，正常读取
                for i in range(jump_number):
                    del self.future_videosize[i]
                #     读取最新推荐视频的video size
                for i in range(jump_number):
                    size_buffer = [[0 for t in range(10)] for k in range(3)]
                    # 如果跳转了两个视频，则按-2，-1顺序读取videosize
                    length = min(10, len(self.net_env.players[-jump_number + i].video_size[0]))
                    for j in range(length):
                        size_buffer[0][j] = self.net_env.players[-jump_number + i].video_size[0][j]
                        size_buffer[1][j] = self.net_env.players[-jump_number + i].video_size[0][j]
                        size_buffer[2][j] = self.net_env.players[-jump_number + i].video_size[0][j]
                    self.future_videosize.append(size_buffer)
            else:  # 推荐列表不足5个，新读入的videosize为0
                for i in range(jump_number):  # 删除无用视频
                    del self.future_videosize[i]
                    # 新数据为0
                    for i in range(jump_number):
                        size_buffer = [[0 for t in range(10)] for k in range(3)]
                        self.future_videosize.append(size_buffer)
            # 4.1 更新条件概率
            # 更新正在播放视频的条件概率
            chunk_remain = self.net_env.players[play_video_id - self.offset].get_remain_video_num()  # 获取剩余块
            playing_chunk = math.ceil(self.net_env.players[play_video_id - self.offset].get_play_chunk())
            for i in range(jump_number):  # 删除无用留存率
                del self.conditional_retent_rate[i]
            if chunk_remain >= 10:
                for i in range(10):
                    # print(self.conditional_retent_rate[play_video_id - self.offset][i])
                    # print(float(self.net_env.players[play_video_id - self.offset].user_retent_rate[(-chunk_remain + i - 1)]))
                    # print(float(self.net_env.players[play_video_id - self.offset].user_retent_rate[playing_chunk]))
                    self.conditional_retent_rate[play_video_id - self.offset][i] = float(
                        self.net_env.players[play_video_id -
                                             self.offset].user_retent_rate[(-chunk_remain + i - 1)]) / \
                                                                                   float(
                                                                                       self.net_env.players[
                                                                                           play_video_id - self.offset].user_retent_rate[
                                                                                           playing_chunk])
            if chunk_remain < 10:  # 不足10块补0
                for i in range(10):
                    if i < chunk_remain:
                        self.conditional_retent_rate[play_video_id - self.offset][i] = \
                            float(self.net_env.players[play_video_id - self.offset].user_retent_rate[
                                      -chunk_remain + i - 1]) / \
                            float(self.net_env.players[play_video_id - self.offset].user_retent_rate[
                                      playing_chunk])
                    else:
                        self.conditional_retent_rate[play_video_id - self.offset][i] = 0

            # 读取新入列推荐视频的条件概率
            if player_length == 5:  # 正常读取新数据
                for i in range(jump_number):
                    conditional_retent_rate_buffer = [0 for t in range(10)]
                    length = min(10, len(self.net_env.players[-jump_number + i].video_size[0]))
                    for j in range(length):
                        conditional_retent_rate_buffer[j] = \
                            self.net_env.players[-jump_number + i].user_retent_rate[j]
                    self.conditional_retent_rate.append(conditional_retent_rate_buffer)
            else:  # 不足5个，新留存率为0
                for i in range(jump_number):
                    conditional_retent_rate_buffer = [0 for t in range(10)]
                    self.conditional_retent_rate.append(conditional_retent_rate_buffer)

            # 5.1 更新缓冲大小
            if player_length == 5:  # 如果推荐列表为满，正常读取
                for i in range(jump_number):
                    self.cache.pop(i)  # 先删除失效cache
                    self.cache.append(0)
                self.cache[play_video_id - self.offset] = rebuf  #
                if download_video_id - self.offset >= 0:  #
                    self.cache[download_video_id - self.offset] = self.net_env.players[
                        download_video_id - self.offset].buffer_size  # 更新下载视频的buff
            else:  # 如果列表不满5个
                if download_video_id - self.offset < 0:  # 说明下载视频已被划走，不更新下载缓冲，只更新播放缓冲
                    for i in range(jump_number):
                        self.cache.pop(i)  # 先删除失效cache
                        self.cache.append(0)
                    self.cache[play_video_id - self.offset] = rebuf
                    print_debug('不满5个且发生滑动的cacha = ' + str(self.cache))
                else:  # 没被划走
                    for i in range(jump_number):
                        self.cache.pop(i)  # 先删除失效cache
                        self.cache.append(0)
                    self.cache[play_video_id - self.offset] = rebuf
                    self.cache[download_video_id - self.offset] = self.net_env.players[
                        download_video_id - self.offset].buffer_size  # 更新下载视频的buff

            # 6.1 更新剩余chunk数
            if player_length == 5:  # 如果推荐列表为满，正常读取
                if download_video_id - self.offset < 0:  # 如果下载视频已经被划走
                    for i in range(jump_number):  # 则更新新推荐列表的视频
                        self.left_chunk_cnt.pop(i)
                        self.left_chunk_cnt.append(self.net_env.players[-jump_number + i].get_remain_video_num())


                else:  # 如果下载视频未被划走，则需要更新下载视频和新读取视频
                    for i in range(jump_number):  # 读取新块数
                        self.left_chunk_cnt.pop(i)
                        self.left_chunk_cnt.append(self.net_env.players[-jump_number + i].get_remain_video_num())
                    self.left_chunk_cnt[download_video_id - self.offset] = self.net_env.players[
                        download_video_id - self.offset].get_remain_video_num()
            else:  # 如果推荐列表不满
                for i in range(jump_number):  # 补0
                    self.left_chunk_cnt.pop(i)
                    self.left_chunk_cnt.append(0)
                # 更新下载块的剩余chunk数
                if download_video_id - self.offset < 0:  # 说明下载视频已被划走，无用
                    pass
                else:
                    self.left_chunk_cnt[download_video_id - self.offset] = self.net_env.players[
                        download_video_id - self.offset].get_remain_video_num()
            print_debug(self.left_chunk_cnt)
            # 7.1 更新上个质量等级
            for i in range(jump_number):
                self.last_5_bitrate.pop(i)
                self.last_5_bitrate.append(0)
            if download_video_id - self.offset >= 0:
                self.last_5_bitrate[download_video_id - self.offset] = bit_rate
            # 去掉播放完的视频
            self.last_play_video_id = play_video_id
            switch_flag = True
            print_debug('state 更新完成')
        # 未发生跳转
        if play_video_id == self.last_play_video_id and switch_flag == False:
            print_debug('当前观看视频仍为' + str(play_video_id) + ',  chunkid' + str(
                math.ceil(self.net_env.players[0].get_play_chunk())))
            # 3.2 更新future_videosize
            # 下载部分
            chunk_remain = self.net_env.players[download_video_id - self.offset].get_remain_video_num()  # 获取剩余块
            if chunk_remain < 10:  # 不足10块补0
                for k in range(3):
                    self.future_videosize[download_video_id - self.offset][k].pop(0)
                    self.future_videosize[download_video_id - self.offset][k].append(0)
            if chunk_remain >= 10:
                for k in range(3):
                    self.future_videosize[download_video_id - self.offset][k].pop(0)
                    self.future_videosize[download_video_id - self.offset][k].append(
                        self.net_env.players[download_video_id - self.offset].video_size[k][9 - chunk_remain])
            # 4.2 更新条件概率
            chunk_play_remain = self.net_env.players[play_video_id - self.offset].get_remain_video_num()
            playing_chunk = math.ceil(self.net_env.players[play_video_id - self.offset].get_play_chunk())
            self.conditional_retent_rate[play_video_id - self.offset] = [0 for t in range(10)]
            if chunk_play_remain >= 10:
                print_debug('>10')
                for i in range(10):
                    self.conditional_retent_rate[play_video_id - self.offset][i] = float(
                        self.net_env.players[0].user_retent_rate[-chunk_play_remain + i - 1]) / \
                                                                                   float(self.net_env.players[
                                                                                             0].user_retent_rate[
                                                                                             playing_chunk])
            if chunk_play_remain < 10:  # 不足10块补0
                print_debug('<10')
                for i in range(10):
                    if i < chunk_play_remain:
                        self.conditional_retent_rate[play_video_id - self.offset][i] = float(
                            self.net_env.players[0].user_retent_rate[-chunk_play_remain + i - 1]) / \
                                                                                       float(self.net_env.players[
                                                                                                 0].user_retent_rate[
                                                                                                 playing_chunk])
                    else:
                        self.conditional_retent_rate[play_video_id - self.offset][i] = 0

            # 5.2 更新缓冲大小
            # 先更新播放部分
            self.cache[play_video_id - self.offset] = rebuf
            # 再更新下载部分
            self.cache[download_video_id - self.offset] = self.net_env.players[
                download_video_id - self.offset].buffer_size
            # 6.2 不切换情况下更新剩余chunk数
            self.left_chunk_cnt[download_video_id - self.offset] = self.net_env.players[
                download_video_id - self.offset].get_remain_video_num()
            # 7.2 更新上个质量等级
            self.last_5_bitrate[download_video_id - self.offset] = bit_rate

        # print(delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes)
        # print log info of the last operation
        if play_video_id < ALL_VIDEO_NUM:
            # the operation results
            current_chunk = self.net_env.players[0].get_play_chunk()
            # print(current_chunk)
            current_bitrate = self.net_env.players[0].get_video_quality(max(int(current_chunk - 1e-10), 0))
        one_step_QOE = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000.

        if play_video_id >= ALL_VIDEO_NUM:  # 全部播放完为done
            # print('结束了')
            done = True

        # 可以输出7个状态
        print_debug('历史吞吐量' + str(self.past_10_throughput))
        print_debug('历史延迟' + str(self.past_10_delay))
        for i in range(5):
            print_debug(self.future_videosize[i])
            print_debug(self.conditional_retent_rate[i])
        print_debug('历史缓冲' + str(self.cache))
        print_debug('历史剩余块数' + str(self.left_chunk_cnt))
        print_debug('历史视频质量' + str(self.last_5_bitrate))
        print_debug('end')
        print_debug("============================================================================================")

        mergelist1 = list(itertools.chain.from_iterable(self.future_videosize))
        mergelist1 = list(itertools.chain.from_iterable(mergelist1))
        mergelist2 = list(itertools.chain.from_iterable(self.conditional_retent_rate))
        mergelist = self.past_10_throughput + self.past_10_delay + mergelist1 + mergelist2 + self.cache + self.left_chunk_cnt + self.last_5_bitrate
        for i in range(235):
            self.state[0][i] = float(mergelist[i])
        for i in range(235):
            if i in range(0, 10):
                self.state[0, i] = self.state[0, i] / 1000000
            if i in range(10, 20):
                self.state[0, i] = self.state[0, i] / 1000
            if i in range(20, 170):
                self.state[0, i] = self.state[0, i] / 1000000
            if i in range(220, 225):
                self.state[0, i] = self.state[0, i] / 1000
            if i in range(225, 230):
                self.state[0, i] = self.state[0, i] / 10
        # return delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes

        return self.state, one_step_QOE, done, flag
