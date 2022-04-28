import sys, os
import glob
import time
from datetime import datetime
import random
import torch
import numpy as np
import gym
from RL_env import RLEnv

from PPO import PPO

sys.path.append('./quickstart/')
import quickstart.fix_preload as Solution

env_name = "test"


################################### Training ###################################
def train():
    print("============================================================================================")
    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = 100  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 10  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = RLEnv()

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    network_batch = 10
    network_dict_size = 20
    user_batch = 20
    user_dict_size = 100
    # training loop
    network_list = range(network_dict_size)
    user_list = range(user_dict_size)
    while time_step <= max_training_timesteps:
        current_ep_reward = 0
        time_step += 1
        for i in random.sample(network_list, network_batch):
            for j in random.sample(user_list, user_batch):
                state = env.reset(i, j)
                done = False
                while done:
                    # select action with policy
                    action = ppo_agent.select_action(state)
                    state, reward, done, _ = env.step(action)

                    # saving reward and is_terminals
                    ppo_agent.buffer.rewards.append(reward)
                    ppo_agent.buffer.is_terminals.append(done)
                    current_ep_reward += reward
        # update PPO agent
        ppo_agent.update()
        print_running_reward += (current_ep_reward / network_batch * user_batch)
        # log in logging file
        # if time_step % log_freq == 0:
        #     # log average reward till last episode
        #     log_avg_reward = log_running_reward / log_running_episodes
        #     log_avg_reward = round(log_avg_reward, 4)
        #
        #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
        #     log_f.flush()
        #
        #     log_running_reward = 0
        #     log_running_episodes = 0
        # printing average reward

        # print average reward till last episode
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                print_running_reward))
        print_running_reward = 0
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        # log_running_reward += current_ep_reward
        # log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


def test():
    # 用于测试state的初始化，更新是否正常
    env = RLEnv()
    solution = Solution.Algorithm()
    solution.Initialize()
    env.reset()
    action = [0 for i in range(3)]
    download_video_id, bit_rate, sleep_time = solution.run(0, 0, 0, False, 0, env.net_env.players, True)
    action[0] = bit_rate
    action[1] = download_video_id
    action[2] = sleep_time
    while True:
        delay, rebuf, video_size, end_of_video, \
        play_video_id, waste_bytes = env.step(action)
        action[1], action[0], action[2] = solution.run(delay, rebuf, video_size, end_of_video, play_video_id,
                                                       env.net_env.players, False)
    print('1')


if __name__ == '__main__':
    test()
