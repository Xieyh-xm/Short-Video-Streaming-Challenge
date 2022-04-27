# import numpy as np
# import fixed_env as env
# import load_trace
# import matplotlib.pyplot as plt
import itertools
from video_player import VIDEO_CHUNCK_LEN
import numpy as np

VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
BITS_IN_BYTE = 8
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
MILLISECONDS_IN_SECOND = 1000.0
MPC_FUTURE_CHUNK=3     #MPC4SHORT
MPC_FUTURE_VIDEO=3
video_deliver=0.05

def sample(past_bandwidth, past_bandwidth_ests, past_errors):
    # ================== MPC =========================
    # shouldn't change the value of past_bandwidth_ests and past_errors in MPC
    copy_past_bandwidth_ests = past_bandwidth_ests
    # print("past bandwidth ests: ", copy_past_bandwidth_ests)
    copy_past_errors = past_errors
    # print("past_errs: ", copy_past_errors)

    curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
    if (len(copy_past_bandwidth_ests) > 0):
        curr_error = abs(copy_past_bandwidth_ests[-1] - past_bandwidth[-1]) / float(past_bandwidth[-1])
    copy_past_errors.append(curr_error)

    # pick bitrate according to MPC
    # first get harmonic mean of last 5 bandwidths
    past_bandwidths = past_bandwidth[-5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]
    # if ( len(state) < 5 ):
    #    past_bandwidths = state[3,-len(state):]
    # else:
    #    past_bandwidths = state[3,-5:]
    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += (1 / float(past_val))
    harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if (len(copy_past_errors) < 5):
        error_pos = -len(copy_past_errors)
    max_error = float(max(copy_past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC here
    future_bandwidth_bps=future_bandwidth*1000000
    # print("future_bd:", future_bandwidth)
    copy_past_bandwidth_ests.append(harmonic_bandwidth)
    return future_bandwidth_bps
    sleep_time=0
    if (buffer_size[0]<=2000):
        video_offset=0
        if(future_bandwidth_bps>=all_future_chunks_size[video_offset][2][0]):
            chunk_q=2
        elif(future_bandwidth_bps>=all_future_chunks_size[video_offset][1][0]):
            chunk_q=1
        else:
            chunk_q=0
    elif((buffer_size[1]<2000) & (len(P)>=2)):
        video_offset=1
        if(future_bandwidth_bps>=all_future_chunks_size[video_offset][2][0]):
            chunk_q=2
        elif(future_bandwidth_bps>=all_future_chunks_size[video_offset][1][0]):
            chunk_q=1
        else:
            chunk_q=0
    elif((buffer_size[2]<2000) & (len(P)==3)):
        video_offset=2
        if(future_bandwidth_bps>=all_future_chunks_size[video_offset][2][0]):
            chunk_q=2
        elif(future_bandwidth_bps>=all_future_chunks_size[video_offset][1][0]):
            chunk_q=1
        else:
            chunk_q=0
    else:
        video_offset=0
        chunk_q=2
        sleep_time=1-all_future_chunks_size[video_offset][2][0]/future_bandwidth_bps
    return  chunk_q,video_offset,sleep_time



    # future chunks length (try 4 if that many remaining)
    # last_index = int(chunk_sum - video_chunk_remain)

    # if ( chunk_sum - last_index < 5 ):
    # future_chunk_length = chunk_sum - last_index

    # start = time.time()
    for combo in CHUNK_COMBO_OPTIONS:
        # combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = np.array(start_buffer,dtype='float32')  # ms
        bitrate_sum = 0
        smoothness_diffs = 0
        # last_quality = int( bit_rate )
        # print(combo)
        lys_curr_buffer = np.array([0,0,0],dtype='float32')
        lys_download_time = 0
        lys_download_size = np.array([0,0,0],dtype='float32')
        total_download_size=0
        for position in range(0, len(combo)):
            chunk_quality = combo[position]%3
            video_offset=int(combo[position]/3)
            # index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = MILLISECONDS_IN_SECOND * (all_future_chunks_size[video_offset][chunk_quality][position] / 1000000.) / (
                future_bandwidth)  # this is MB/MB/s --> seconds
            # print("download time:", MILLISECONDS_IN_SECOND, "*",  (all_future_chunks_size[chunk_quality][position]/1000000.), "/", future_bandwidth)
            # lys test
            lys_curr_buffer+=curr_buffer
            lys_download_time+=download_time
            watchtime_chunk=int(lys_download_time/1000.0+1)
            lys_download_size[video_offset]+=(all_future_chunks_size[video_offset][chunk_quality][position])
            if (curr_buffer[0] < download_time):
                curr_rebuffer_time += (download_time-curr_buffer[0])*(1-watchtime_chunk*video_deliver)
                curr_buffer[0] = 0
            else:
                curr_buffer[0] -= download_time
            if(curr_buffer[1]<download_time):
                curr_rebuffer_time += download_time*watchtime_chunk*video_deliver
            if(curr_buffer[2]<download_time):
                curr_rebuffer_time += download_time * watchtime_chunk * video_deliver* watchtime_chunk * video_deliver

            curr_buffer[video_offset] += VIDEO_CHUNCK_LEN
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]*conditional_retent[video_offset][position]
            smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])\
                                *conditional_retent[video_offset][position]
            last_quality = chunk_quality
            total_download_size+=all_future_chunks_size[video_offset][chunk_quality][position]
        reward = (bitrate_sum / 1000.) - (REBUF_PENALTY * curr_rebuffer_time / 1000.) - (smoothness_diffs / 1000.)-(total_download_size*4/1000000.)
        # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)
        if (reward >= max_reward):
            if (best_combo != ()) and best_combo[0] < combo[0]:
                best_combo = combo
            else:
                best_combo = combo

            max_reward = reward
            lys_rebuf = curr_rebuffer_time
            lys_combo = combo
            send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
            if (best_combo != ()):  # some combo was good
                send_data = best_combo[0]

    bit_rate = send_data%3
    video_id=  int(send_data/3)
    return bit_rate, video_id