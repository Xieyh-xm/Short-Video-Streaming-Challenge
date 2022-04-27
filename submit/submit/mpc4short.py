# Comparison Algorithm: No saving approach
# No saving algorithm downloads the current playing video first.
# When the current playing video download ends, it preloads the videos on the recommendation queue periodically, with 800KB for each video.

import numpy as np
import sys
sys.path.append("..")
from simulator.video_player import BITRATE_LEVELS
from simulator import mpc_module4short
from simulator import sample

MPC_FUTURE_CHUNK_COUNT = 3     # MPC
MPC_FUTURE_VIDEO=3    #MPC4SHORT
PAST_BW_LEN = 5
TAU = 500.0  # ms
RECOMMEND_QUEUE = 5  
PROLOAD_SIZE = 800000.0   # B

class Algorithm:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        self.sleep_time = 0

    # Intial
    def Initialize(self):
        # Initialize your session or something
        # past bandwidth record
        self.past_bandwidth = np.zeros(PAST_BW_LEN)

    def estimate_bw(self):
        # record the newest error
        curr_error = 0  # default assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(self.past_bandwidth_ests) > 0) and self.past_bandwidth[-1] != 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidth[-1])/float(self.past_bandwidth[-1])
        self.past_errors.append(curr_error)
        # first get harmonic mean of last 5 bandwidths
        past_bandwidth = self.past_bandwidth[:]
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:]
        bandwidth_sum = 0
        for past_val in past_bandwidth:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidth))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(self.past_errors) < 5 ):
            error_pos = -len(self.past_errors)
        max_error = float(max(self.past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        self.past_bandwidth_ests.append(harmonic_bandwidth)
        # self.past_bandwidth = np.roll(self.past_bandwidth, -1)
        # self.past_bandwidth[-1] = future_bandwidth

    # Define your algorithm
    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        DEFAULT_QUALITY = 0
        if first_step:   # 第一步没有任何信息
            self.sleep_time = 0
            return 0, 2, self.sleep_time

        # download a chunk, record the bitrate and update the network 
        if self.sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size)/1000000.0) /(float(delay) / 1000.0)  # MB / s
        
        P = []
        all_future_chunks_size = []
        future_chunks_highest_size = []
        offset=0
        buffer_size=[]
        for i in range(min(len(Players), RECOMMEND_QUEUE)):
            if Players[i].get_remain_video_num() == 0:      # download over
                P.append(0)
                all_future_chunks_size.append([0])
                future_chunks_highest_size.append([0])
                offset+=1
                continue
            
            P.append(Players[i].get_remain_video_num())
            all_future_chunks_size.append(Players[i].get_undownloaded_video_size(P[-1]))
            future_chunks_highest_size.append(all_future_chunks_size[-1][BITRATE_LEVELS-1])
            buffer_size.append(Players[i].get_buffer_size())
        sum=0
        for i in range(len(P)):
            sum+=P[i]
        if sum==0:
            return play_video_id,0,0.0
        # update past_errors and past_bandwidth_ests
        self.estimate_bw()
        future_bandwidth=sample.sample(self.past_bandwidth, self.past_bandwidth_ests,self.past_errors)
        buffer_offset=0
        length=len(buffer_size)
        download_video_id = play_video_id + offset
        self.sleep_time=0.0
        if length==1:
            if(buffer_size[0]>2000):
                download_video_id=play_video_id+offset
                self.sleep_time=250.0
                return download_video_id,2,250
            else:
                download_video_id = play_video_id + offset
        if(length==2 ):
            if(buffer_size[0]>2000)&(buffer_size[1]>=2000):
                download_video_id = play_video_id + offset
                self.sleep_time = 500.0
                return download_video_id, 2, 250
            elif (buffer_size[0]>=2000):
                download_video_id=1+play_video_id+offset
            else:
                download_video_id = play_video_id + offset
        if(length>=3):
            if ((buffer_size[0]>2000) & (buffer_size[1]>=2000) & (buffer_size[2]>=2000)):
                download_video_id = play_video_id + offset
                self.sleep_time = 500.0
                return download_video_id, 2, 250
            elif((buffer_size[0]>2000) & (buffer_size[1]>=2000)):
                download_video_id=2+play_video_id+offset
            elif(buffer_size[0]>2000):
                download_video_id = 1 + play_video_id + offset
            else:
                download_video_id = play_video_id + offset

        if (future_bandwidth >= all_future_chunks_size[download_video_id-play_video_id][2][0]):
            chunk_q = 2
        elif (future_bandwidth >= all_future_chunks_size[download_video_id-play_video_id][1][0]):
            chunk_q = 1
        else:
            chunk_q = 0


        # print("choosing bitrate for: ", download_video_id, ", chunk: ", Players[download_video_seq].get_chunk_counter())
        # print("past_bandwidths:", self.past_bandwidth[-5:], "past_ests:", self.past_bandwidth_ests[-5:])

        # bit_rate,download_video_offset = mpc_module4short.mpc(self.past_bandwidth, self.past_bandwidth_ests, self.past_errors,
        #                                 all_future_chunks_size[play_video_id : play_video_id+MPC_FUTURE_VIDEO],
        #                                 P[play_video_id:play_video_id+MPC_FUTURE_VIDEO], buffer_size, chunk_sum,
        #                                 video_chunk_remain, last_quality,Players,play_video_id)

        # bit_rate, download_video_offset,sleep_time = sample.sample(self.past_bandwidth, self.past_bandwidth_ests,
        #                                                        self.past_errors,
        #                                                        all_future_chunks_size[
        #                                                        play_video_id: play_video_id + MPC_FUTURE_VIDEO],
        #                                                        P[play_video_id:play_video_id + MPC_FUTURE_VIDEO],
        #                                                        buffer_size, chunk_sum,
        #                                                        video_chunk_remain, last_quality, Players,
        #                                                        play_video_id)

        return download_video_id, chunk_q, self.sleep_time