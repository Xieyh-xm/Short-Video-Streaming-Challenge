import torch
import math
import itertools
from results.PPO_sleep_update import PPO
import collections

# NN_MODEL = "/home/team/" + "ParttimeJob" + "/submit/results/PPO_mix_train_0_350.pth"
NN_MODEL = "submit/submit/results/PPO_update_throughput_0_475.pth"

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

TAU = 500

STATE_NUM = 30
ACTION_NUM = 16


class Algorithm:
    def __init__(self):
        # fill your self params
        self.ppo_agent = PPO(STATE_NUM, ACTION_NUM, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                             has_continuous_action_space,
                             action_std)
        self.ppo_agent.load(NN_MODEL)

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
        self.update_time = 0
        self.playtime_line = 0

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

        self.update_time = 0
        self.playtime_line = 0

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
        self.playtime_line += delay

        # 1. 判断网络状况
        # throughput = 8 * (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # Mbps

        # 2. rl读入状态
        if first_step:
            self.calculate_state(delay, video_size, play_video_id, Players, first_step)
            self.last_sleep_time = 0
            self.last_bitrate = 0
            return 0, 0, 0
        self.calculate_state(delay, video_size, play_video_id, Players, first_step)

        # 3. rl决策
        action = self.ppo_agent.select_action(self.newstate)
        self.download_video_id = int(action[0])
        self.download_video_id += self.offset
        bit_rate = int(action[1])
        if action[2]:
            sleep_time = TAU
        else:
            sleep_time = 0
        self.last_sleep_time = sleep_time
        self.last_bitrate = bit_rate

        print('当前播放视频id ', play_video_id, '当前决策为下载视频', self.download_video_id, '，sleep_tiem=', sleep_time, '码率等级为',
              bit_rate, 'throughput',self.newstate[0, 0:5])
        return self.download_video_id, bit_rate, sleep_time

    def calculate_state(self, delay, video_size, play_video_id, Players, first_step=False):
        switch_flag = False
        sleep_flag = False
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

            if not sleep_flag:
                if self.playtime_line - self.update_time >= 1000.:
                    self.past_10_throughput.pop(0)
                    self.past_10_throughput.append(video_size * 8 / delay * 1000.0)
                    self.update_time = self.playtime_line

            # 2.过去的delay
            # self.past_10_delay = numpy.delete(self.past_10_delay, 0)
            # self.past_10_delay = numpy.append(self.past_10_delay, delay)
            if not sleep_flag:
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

        self.newstate[:, 0:5] = self.state[:, 5:10]
        index = torch.tensor([20, 50, 80, 110, 140])
        self.newstate[:, 5:10] = self.state[:, index.long()]
        index = torch.tensor([170, 180, 190, 200, 210])
        self.newstate[:, 10:15] = self.state[:, index.long()]
        self.newstate[:, 15:30] = self.state[:, 220:235]

        # self.newstate[:, 0:10] = self.state[:, 0:10]
        # index = torch.tensor([20, 50, 80, 110, 140])
        # self.newstate[:, 10:15] = self.state[:, index.long()]
        # index = torch.tensor([170, 180, 190, 200, 210])
        # self.newstate[:, 15:20] = self.state[:, index.long()]
        # self.newstate[:, 20:35] = self.state[:, 220:235]
