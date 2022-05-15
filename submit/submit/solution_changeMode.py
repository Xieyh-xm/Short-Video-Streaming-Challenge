import torch
import math
import itertools
import collections
import numpy as np
from results.PPO_sleep_general import PPO as PPO_general
from results.PPO_sleep_High_Medium import PPO as PPO_high_medium
from math import log

# NN_MODEL_general = "/home/team/" + "ParttimeJob" + "/submit/results/PPO_20220510_0_400.pth"
# NN_MODEL_High_Medium = "/home/team/" + "ParttimeJob" + "/submit/results/PPO_high.pth"
NN_MODEL_general = "submit/submit/results/PPO_20220510_0_400.pth"
NN_MODEL_High_Medium = "submit/submit/results/PPO_high.pth"

lr_actor = 0.0001  # learning rate for actor network
lr_critic = 0.0005  # learning rate for critic network
K_epochs = 80  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.85  # discount factor
has_continuous_action_space = False
action_std = 0.6

STATE_DIMENSION = 1
HISTORY_LENGTH = 235
# ALL_VIDEO_NUM = 7
ALL_VIDEO_NUM = 9
VIDEO_BIT_RATE = [750, 1200, 1850]

TAU = 250

STATE_NUM = 35
ACTION_NUM = 16

high = [0.0, 0.023095238095238096, 0.003545918367346939, 0.00413265306122449, 0.004336734693877551,
        0.00510204081632653, 0.0054761904761904765, 0.006420068027210885, 0.007321428571428572,
        0.008095238095238095, 0.0091156462585034, 0.00939625850340136, 0.010391156462585035,
        0.011904761904761904, 0.01191326530612245, 0.012959183673469387, 0.014056122448979592,
        0.01573979591836735, 0.01608843537414966, 0.01778061224489796, 0.017831632653061223,
        0.01897108843537415, 0.019753401360544216, 0.02094387755102041, 0.021181972789115645,
        0.022193877551020407, 0.022474489795918366, 0.022363945578231292, 0.023112244897959183,
        0.022704081632653063, 0.02312074829931973, 0.022406462585034013, 0.02312074829931973,
        0.022899659863945578, 0.022831632653061224, 0.021700680272108842, 0.021462585034013604,
        0.021445578231292516, 0.020450680272108845, 0.020144557823129253, 0.019889455782312927,
        0.019098639455782312, 0.018316326530612246, 0.017465986394557823, 0.016760204081632653,
        0.016377551020408162, 0.0158078231292517, 0.014336734693877551, 0.014056122448979592,
        0.013741496598639456, 0.012993197278911565, 0.012704081632653061, 0.011352040816326531,
        0.010892857142857143, 0.010391156462585035, 0.009668367346938775, 0.009506802721088435,
        0.008801020408163265, 0.008545918367346939, 0.007678571428571429, 0.007644557823129252,
        0.007806122448979592, 0.0064285714285714285, 0.006207482993197279, 0.005833333333333334,
        0.004957482993197279, 0.004770408163265306, 0.0048384353741496595, 0.004472789115646259,
        0.004064625850340136, 0.004022108843537415, 0.003698979591836735, 0.003486394557823129,
        0.002933673469387755, 0.0027380952380952383, 0.0025340136054421768, 0.0026785714285714286,
        0.002066326530612245, 0.0021258503401360546, 0.0018027210884353742, 0.0016666666666666668,
        0.0014370748299319727, 0.0013435374149659864, 0.0012670068027210884, 0.001284013605442177,
        0.001096938775510204, 0.0009523809523809524, 0.0009608843537414966, 0.0009183673469387755,
        0.0007482993197278912, 0.0007142857142857143, 0.000586734693877551, 0.0005697278911564625,
        0.0005102040816326531, 0.0005442176870748299, 0.0005187074829931973, 0.0003401360544217687,
        0.00040816326530612246, 0.00045068027210884356]
low_and_medium = [0.0, 0.00790391156462585, 0.004481292517006802, 0.00814625850340136, 0.014804421768707483,
                  0.02982142857142857, 0.05215561224489796, 0.06863945578231292, 0.07489795918367347,
                  0.07439625850340136, 0.07158163265306122, 0.06543367346938775, 0.060625, 0.05586309523809524,
                  0.05059948979591837, 0.04553996598639456, 0.04144132653061224, 0.036326530612244896,
                  0.031785714285714285, 0.028154761904761905, 0.024880952380952382, 0.021641156462585034,
                  0.018732993197278912, 0.016364795918367345, 0.01370748299319728, 0.01199829931972789,
                  0.010259353741496599, 0.008966836734693878, 0.007699829931972789, 0.006738945578231292,
                  0.0055824829931972785, 0.004995748299319728, 0.004396258503401361, 0.003660714285714286,
                  0.003048469387755102, 0.0025935374149659864, 0.002134353741496599, 0.0019217687074829931,
                  0.0016496598639455782, 0.001135204081632653, 0.0009778911564625851, 0.0008290816326530612,
                  0.0007270408163265307, 0.0005229591836734694, 0.00045918367346938773, 0.0003401360544217687,
                  0.0003316326530612245, 0.00025935374149659864, 0.00019982993197278912, 0.00013605442176870748,
                  0.00014880952380952382, 9.77891156462585e-05, 5.5272108843537416e-05, 3.401360544217687e-05,
                  3.826530612244898e-05, 3.401360544217687e-05, 1.7006802721088435e-05, 2.9761904761904762e-05,
                  1.7006802721088435e-05, 1.7006802721088435e-05, 8.503401360544217e-06, 4.251700680272109e-06,
                  0.0, 8.503401360544217e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0]
low = [0.0, 0.005501700680272109, 0.004183673469387755, 0.008341836734693877, 0.016819727891156464,
       0.03506802721088435, 0.06459183673469387, 0.08556972789115647, 0.09490646258503402, 0.09329931972789116,
       0.08622448979591837, 0.07691326530612246, 0.06915816326530612, 0.06136904761904762, 0.05282312925170068,
       0.04733843537414966, 0.04000850340136054, 0.033188775510204084, 0.02744047619047619,
       0.021462585034013604, 0.01866496598639456, 0.014379251700680272, 0.011658163265306122,
       0.008520408163265306, 0.006224489795918367, 0.004965986394557823, 0.0035289115646258504,
       0.002551020408163265, 0.0017346938775510204, 0.001173469387755102, 0.0006972789115646258,
       0.0005272108843537415, 0.0004166666666666667, 0.00031462585034013605, 0.00016156462585034014,
       0.00010204081632653062, 4.2517006802721085e-05, 5.102040816326531e-05, 2.5510204081632654e-05,
       1.7006802721088435e-05, 8.503401360544217e-06, 1.7006802721088435e-05, 8.503401360544217e-06, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
medium = [0.0, 0.010306122448979592, 0.004778911564625851, 0.007950680272108844, 0.012789115646258504,
          0.02457482993197279, 0.03971938775510204, 0.05170918367346939, 0.054889455782312926,
          0.05549319727891156, 0.056938775510204084, 0.05395408163265306, 0.05209183673469388,
          0.05035714285714286, 0.048375850340136056, 0.043741496598639455, 0.04287414965986395,
          0.039464285714285716, 0.03613095238095238, 0.034846938775510206, 0.031096938775510206,
          0.028903061224489796, 0.0258078231292517, 0.024209183673469387, 0.02119047619047619,
          0.01903061224489796, 0.016989795918367346, 0.01538265306122449, 0.013664965986394557,
          0.012304421768707482, 0.010467687074829932, 0.009464285714285715, 0.008375850340136054,
          0.007006802721088435, 0.005935374149659864, 0.005085034013605442, 0.004226190476190476,
          0.003792517006802721, 0.003273809523809524, 0.0022534013605442177, 0.0019472789115646259,
          0.001641156462585034, 0.001445578231292517, 0.0010459183673469387, 0.0009183673469387755,
          0.0006802721088435374, 0.000663265306122449, 0.0005187074829931973, 0.00039965986394557823,
          0.00027210884353741496, 0.00029761904761904765, 0.000195578231292517, 0.00011054421768707483,
          6.802721088435374e-05, 7.653061224489796e-05, 6.802721088435374e-05, 3.401360544217687e-05,
          5.9523809523809524e-05, 3.401360544217687e-05, 3.401360544217687e-05, 1.7006802721088435e-05,
          8.503401360544217e-06, 0.0, 1.7006802721088435e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0]


class Algorithm:
    def __init__(self):
        # ======== rl模型 ========
        self.past_throughput = collections.deque(maxlen=10)
        self.play_timeline = 0
        self.update_timeline = 0
        self.choose_network = None
        # 通用模型
        self.ppo_agent_general = PPO_general(STATE_NUM, ACTION_NUM, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                                             has_continuous_action_space,
                                             action_std)
        self.ppo_agent_general.load(NN_MODEL_general)
        # 中高带宽模型
        self.ppo_agent_High_Medium = PPO_high_medium(STATE_NUM, ACTION_NUM, lr_actor, lr_critic, gamma, K_epochs,
                                                     eps_clip,
                                                     has_continuous_action_space,
                                                     action_std)
        self.ppo_agent_High_Medium.load(NN_MODEL_High_Medium)
        self.state = torch.zeros(STATE_DIMENSION, HISTORY_LENGTH)
        self.newstate = torch.zeros(1, STATE_NUM)

        # ======== extra param ========
        self.last_bitrate = 0
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
        self.download_video_id = 0
        self.last_sleep_time = 0  # 决策上次睡眠时间

    # Intial
    def Initialize(self):
        # Initialize your session or something
        # state 状态数组
        self.left_chunk_cnt = []  # 剩余chunk数
        # self.state = torch.zeros((1, STATE_DIMENSION, HISTORY_LENGTH))
        self.last_bitrate = 0
        self.last_play_video_id = 0
        self.last_play_chunk_idx = -1
        self.last_residual_time = 0
        # 填充state
        self.past_10_throughput = [1000000 for i in range(10)]
        self.past_10_delay = [1000 for i in range(10)]
        self.cache = [0 for i in range(5)]
        self.last_5_bitrate = [0 for i in range(5)]
        self.future_videosize = [[[0 for t in range(10)] for i in range(3)] for k in range(5)]

        self.conditional_retent_rate = [[0 for t in range(10)] for k in range(5)]
        self.left_chunk_cnt = [0 for i in range(5)]
        self.offset = 0
        self.state = torch.zeros(STATE_DIMENSION, HISTORY_LENGTH)
        self.newstate = torch.zeros(1, STATE_NUM)
        self.download_video_id = 0

    # Define your algorithm
    # The args you can get are as follows:
    # 1. delay: the time cost of your last operation
    # 2. rebuf: the length of rebufferment
    # 3. video_size: the size of the last downloaded chunk
    # 4. end_of_video: if the last video was ended
    # 5. play_video_id: the id of the current video
    # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
    # 7. first_step: is this your first step?
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        self.play_timeline += delay  # 播放时间线
        # 1. rl读入状态
        if first_step:
            self.calculate_state(delay, video_size, play_video_id, Players, first_step)
            self.last_sleep_time = 0
            self.last_bitrate = 0
            return 0, 0, 0

        # 2. 判断网络状况
        if delay != 0 and self.last_sleep_time == 0:
            throughput = 8 * (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # Mbps
            self.past_throughput.append(throughput)

        # ============= 计算KL散度 =============
        KL_high = 0.0
        KL_medium = 0.0
        KL_low = 0.0

        if len(self.past_throughput) >= 3:
            for i, val in enumerate(self.past_throughput):
                idx = int(val * 10)
                p = 1 / len(self.past_throughput)
                KL_high += p * log(p / high[idx], 2)
                KL_medium += p * log(p / medium[idx], 2)
                KL_low += p * log(p / low[idx], 2)

        # if not self.choose_network and self.play_timeline >= 6000.:
        #     avg_throughput = sum(self.past_throughput) / len(self.past_throughput)
        #     if avg_throughput > 1.6:
        #         self.choose_network = "medium or high"
        #         # print("high")
        #     else:
        #         self.choose_network = "low"
        #         # print("low")
        if KL_low > KL_high and KL_low > KL_medium:
            self.choose_network = "low"
        else:
            self.choose_network = "medium or high"

        # 3. rl决策
        self.calculate_state(delay, video_size, play_video_id, Players, first_step)
        if not self.choose_network:  # 还没决策
            action = self.ppo_agent_general.select_action(self.newstate)
        elif self.choose_network == "medium or high":
            action = self.ppo_agent_High_Medium.select_action(self.newstate)
        else:  # low
            action = self.ppo_agent_general.select_action(self.newstate)

        self.download_video_id = int(action[0])
        self.download_video_id += self.offset
        bit_rate = int(action[1])
        if action[2]:
            sleep_time = TAU
        else:
            sleep_time = 0
        self.last_sleep_time = sleep_time
        self.last_bitrate = bit_rate

        # print('当前播放视频id ', play_video_id, '当前决策为下载视频', self.download_video_id, '，sleep_tiem=', sleep_time, '码率等级为',
        #       bit_rate, 'buffer',
        #       self.newstate[0, 20:25])
        return self.download_video_id, bit_rate, sleep_time

    def calculate_state(self, delay, video_size, play_video_id, Players, first_step=False):
        sleep_flag = False
        switch_flag = False
        if first_step:
            # 获取五个视频三个质量未来10个chunk videosize
            i = 0
            for player in Players:
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
        else:
            # 检查player长度
            player_length = len(Players)
            if self.last_sleep_time > 0:
                sleep_flag = True
            # 获取state状态
            # 1.过去的吞吐量
            # self.past_10_throughput = numpy.delete(self.past_10_throughput, 0)
            # self.past_10_throughput = numpy.append(self.past_10_throughput, video_size)
            if sleep_flag:
                pass
            else:
                self.past_10_throughput.pop(0)
                self.past_10_throughput.append(video_size * 8 / delay * 1000.0)
            # 2.过去的delay
            # self.past_10_delay = numpy.delete(self.past_10_delay, 0)
            # self.past_10_delay = numpy.append(self.past_10_delay, delay)
            if sleep_flag:
                pass
            else:
                self.past_10_delay.pop(0)
                self.past_10_delay.append(delay)
            # 如果发生视频跳转 需要更新新播放视频和新下载视频的数据
            # play_video_id,download_id = 0-6 此时视频id已经+1 ，

            if play_video_id > self.last_play_video_id and play_video_id < ALL_VIDEO_NUM:
                jump_number = play_video_id - self.last_play_video_id  # 一次跳转的视频个数
                self.offset += jump_number
                # 3.1 更新future_videosize
                if player_length == 5:  # 如果推荐列表为满，正常读取
                    for i in range(jump_number):
                        del self.future_videosize[0]
                    #     读取最新推荐视频的video size
                    for i in range(jump_number):
                        size_buffer = [[0 for t in range(10)] for k in range(3)]
                        # 如果跳转了两个视频，则按-2，-1顺序读取videosize
                        length = min(10, len(Players[-jump_number + i].video_size[0]))
                        for j in range(length):
                            size_buffer[0][j] = Players[-jump_number + i].video_size[0][j]
                            size_buffer[1][j] = Players[-jump_number + i].video_size[1][j]
                            size_buffer[2][j] = Players[-jump_number + i].video_size[2][j]
                        self.future_videosize.append(size_buffer)
                else:  # 推荐列表不足5个，新读入的videosize为0
                    for i in range(jump_number):  # 删除无用视频
                        del self.future_videosize[0]
                        # 新数据为0
                    for i in range(jump_number):
                        size_buffer = [[0 for t in range(10)] for k in range(3)]
                        self.future_videosize.append(size_buffer)
                # 更新当前下载视频的未来10个size
                if self.download_video_id - self.offset >= 0 and not sleep_flag:  # 下载的视频未被划走
                    chunk_remain = Players[self.download_video_id - self.offset].get_remain_video_num()  # 获取剩余块
                    if chunk_remain < 10:  # 不足10块补0
                        for k in range(3):
                            self.future_videosize[self.download_video_id - self.offset][k].pop(0)
                            self.future_videosize[self.download_video_id - self.offset][k].append(0)
                    if chunk_remain >= 10:
                        for k in range(3):
                            self.future_videosize[self.download_video_id - self.offset][k].pop(0)
                            self.future_videosize[self.download_video_id - self.offset][k].append(
                                Players[self.download_video_id - self.offset].video_size[k][9 - chunk_remain])

                # 4.1 更新条件概率
                # 更新正在播放视频的条件概率
                chunk_play_remain = Players[0].get_remain_video_num()  # 获取剩余块
                playing_chunk = math.ceil(Players[0].get_play_chunk())
                for i in range(jump_number):  # 删除无用留存率
                    del self.conditional_retent_rate[0]
                if chunk_play_remain >= 10:
                    for i in range(10):
                        self.conditional_retent_rate[0][i] = float(
                            Players[0].user_retent_rate[-chunk_play_remain + i - 2]) / float(
                            Players[0].user_retent_rate[max(0, playing_chunk - 1)])
                if chunk_play_remain < 10:  # 不足10块补0
                    for i in range(10):
                        if i < chunk_play_remain:
                            self.conditional_retent_rate[0][i] = \
                                float(Players[0].user_retent_rate[
                                          -chunk_play_remain + i - 2]) / \
                                float(Players[0].user_retent_rate[max(0, playing_chunk - 1)])
                        else:
                            self.conditional_retent_rate[0][i] = 0

                # 读取新入列推荐视频的条件概率
                cur_length = 0  # 实际留存率中视频个数
                cur_length = len(self.conditional_retent_rate)
                diff = player_length - cur_length  # 实际剩余视频个数减去state中有的视频个数
                for i in range(diff):
                    conditional_retent_rate_buffer = [0 for t in range(10)]
                    length = min(10, len(Players[cur_length + i].video_size[0]))
                    for j in range(length):
                        conditional_retent_rate_buffer[j] = Players[cur_length + i].user_retent_rate[j]
                    self.conditional_retent_rate.append(conditional_retent_rate_buffer)

                cur_length = len(self.conditional_retent_rate)
                for i in range(5 - cur_length):  # 如果还不够补0
                    conditional_retent_rate_buffer = [0 for t in range(10)]
                    self.conditional_retent_rate.append(conditional_retent_rate_buffer)

                # 读取正在下载的留存率
                if self.download_video_id - self.offset > 0 and not sleep_flag:
                    if chunk_remain >= 10:  # 直接读
                        for j in range(10):
                            self.conditional_retent_rate[self.download_video_id - self.offset][j] = \
                                Players[self.download_video_id - self.offset].user_retent_rate[(-chunk_remain + 1) + j]
                    if chunk_remain < 10:
                        self.conditional_retent_rate[self.download_video_id - self.offset].pop(0)
                        self.conditional_retent_rate[self.download_video_id - self.offset].append(0)

                # 5.1 更新缓冲大小
                for i in range(jump_number):
                    self.cache.pop(0)  # 先删除失效cache
                    if self.last_play_video_id + 4 + jump_number < ALL_VIDEO_NUM:
                        self.cache.append(0)
                    else:
                        self.cache.append(10000)
                if self.download_video_id - self.offset >= 0:  # 满足如果下载视频未被划走
                    if not sleep_flag:
                        self.cache[self.download_video_id - self.offset] = Players[
                            self.download_video_id - self.offset].buffer_size  # 更新下载视频的buff
                    self.cache[0] = Players[0].buffer_size  # 更新播放视频的buff
                if self.download_video_id - self.offset < 0:  # 说明下载视频已被划走，不更新下载缓冲，只更新播放缓冲
                    self.cache[0] = Players[0].buffer_size
                    # print_debug('不满5个且发生滑动的cacha = ' + str(self.cache))
                # 6.1 更新剩余chunk数
                for i in range(5):
                    if i >= player_length:
                        self.left_chunk_cnt[i] = 0
                    else:
                        self.left_chunk_cnt[i] = Players[i].get_remain_video_num()
                # 7.1 更新上个质量等级
                for i in range(jump_number):
                    self.last_5_bitrate.pop(0)
                    self.last_5_bitrate.append(0)
                if self.download_video_id - self.offset >= 0 and not sleep_flag:
                    self.last_5_bitrate[self.download_video_id - self.offset] = self.last_bitrate
                # 去掉播放完的视频
                self.last_play_video_id = play_video_id
                switch_flag = True
            # 未发生跳转
            if play_video_id == self.last_play_video_id and not switch_flag:
                # 3.2 更新future_videosize
                # 下载部分
                chunk_remain = Players[self.download_video_id - self.offset].get_remain_video_num()  # 获取剩余块
                if not sleep_flag:
                    if chunk_remain < 10:  # 不足10块补0
                        for k in range(3):
                            self.future_videosize[self.download_video_id - self.offset][k].pop(0)
                            self.future_videosize[self.download_video_id - self.offset][k].append(0)
                    if chunk_remain >= 10:
                        for k in range(3):
                            self.future_videosize[self.download_video_id - self.offset][k].pop(0)
                            self.future_videosize[self.download_video_id - self.offset][k].append(
                                Players[self.download_video_id - self.offset].video_size[k][9 - chunk_remain])
                # 4.2 更新条件概率
                # 更新正在播放视频的条件概率
                chunk_play_remain = Players[0].get_remain_video_num()
                playing_chunk = math.ceil(Players[0].get_play_chunk())
                self.conditional_retent_rate[0] = [0 for t in range(10)]
                if chunk_play_remain >= 10:
                    for i in range(10):
                        self.conditional_retent_rate[0][i] = float(
                            Players[0].user_retent_rate[-chunk_play_remain + i - 2]) / \
                                                             float(Players[
                                                                       0].user_retent_rate[
                                                                       playing_chunk])
                if chunk_play_remain < 10:  # 不足10块补0
                    for i in range(10):
                        if i < chunk_play_remain:
                            self.conditional_retent_rate[0][i] = float(
                                Players[0].user_retent_rate[-chunk_play_remain + i - 2]) / \
                                                                 float(Players[
                                                                           0].user_retent_rate[
                                                                           playing_chunk])
                        else:
                            self.conditional_retent_rate[0][i] = 0

                if self.download_video_id - self.offset > 0 and not sleep_flag:
                    if chunk_remain >= 10:  # 直接读
                        for j in range(10):
                            self.conditional_retent_rate[self.download_video_id - self.offset][j] = float(
                                Players[self.download_video_id - self.offset].user_retent_rate[(-chunk_remain + 1) + j])
                if chunk_remain < 10:
                    self.conditional_retent_rate[self.download_video_id - self.offset].pop(0)
                    self.conditional_retent_rate[self.download_video_id - self.offset].append(0)

                # 5.2 更新缓冲大小
                # 先更新播放部分
                if not sleep_flag:
                    self.cache[self.download_video_id - self.offset] = Players[
                        self.download_video_id - self.offset].buffer_size
                self.cache[0] = Players[
                    0].buffer_size
                # 6.2 不切换情况下更新剩余chunk数
                if not sleep_flag:
                    self.left_chunk_cnt[self.download_video_id - self.offset] = Players[
                        self.download_video_id - self.offset].get_remain_video_num()
                # 7.2 更新上个质量等级
                if not sleep_flag:
                    self.last_5_bitrate[self.download_video_id - self.offset] = self.last_bitrate

            # print(delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes)
            # print log info of the last operation
            if play_video_id < ALL_VIDEO_NUM:
                # the operation results
                current_chunk = Players[0].get_play_chunk()
                # print(current_chunk)
                current_bitrate = Players[0].get_video_quality(max(int(current_chunk - 1e-10), 0))

            if play_video_id >= ALL_VIDEO_NUM:  # 全部播放完为done
                # print('结束了')
                done = True

        mergelist1 = list(itertools.chain.from_iterable(self.future_videosize))
        mergelist1 = list(itertools.chain.from_iterable(mergelist1))
        mergelist2 = list(itertools.chain.from_iterable(self.conditional_retent_rate))
        mergelist = self.past_10_throughput + self.past_10_delay + mergelist1 + mergelist2 + self.cache + self.left_chunk_cnt + self.last_5_bitrate
        for i in range(235):
            self.state[0][i] = float(mergelist[i])

        self.state[0, 0:10] = self.state[0, 0:10] / 1000000
        self.state[0, 10:20] = self.state[0, 10:20] / 1000
        self.state[0, 20:170] = self.state[0, 20:170] / 1000000
        self.state[0, 220:225] = self.state[0, 220:225] / 1000
        self.state[0, 225:230] = self.state[0, 225:230] / 10

        # self.newstate[:, 0:5] = self.state[:, 5:10]
        # index = torch.tensor([20, 50, 80, 110, 140])
        # self.newstate[:, 5:10] = self.state[:, index.long()]
        # index = torch.tensor([170, 180, 190, 200, 210])
        # self.newstate[:, 10:15] = self.state[:, index.long()]
        # self.newstate[:, 15:30] = self.state[:, 220:235]

        self.newstate[:, 0:10] = self.state[:, 0:10]
        index = torch.tensor([20, 50, 80, 110, 140])
        self.newstate[:, 10:15] = self.state[:, index.long()]
        index = torch.tensor([170, 180, 190, 200, 210])
        self.newstate[:, 15:20] = self.state[:, index.long()]
        self.newstate[:, 20:35] = self.state[:, 220:235]
