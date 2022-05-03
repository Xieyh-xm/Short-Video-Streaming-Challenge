import os
import glob
import time
from datetime import datetime
import random
import torch
import numpy as np
import gym
from RL_env import RLEnv
from DQN import DQN
from config import Config

env_name = "DQN-gym"


################################### Training ###################################
def train():
    print("============================================================================================")
    has_continuous_action_space = False  # continuous action space; else discrete
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps
    save_model_freq = 5  # save model frequency (in num timesteps)
    random_seed = 4

    #####################################################

    print("training environment name : " + env_name)

    env = RLEnv()

    # state space dimension
    # state_dim = 235
    # state_dim = 55
    state_dim = 35
    action_dim = 15

    # action space dimension
    # if has_continuous_action_space:
    #     action_dim = env.action_space.shape[0]
    # else:
    #     action_dim = env.action_space.n

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = "DQN_logs"
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
    log_f_name = log_dir + 'DQN_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "DQN_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a DQN agent
    cfg = Config()
    DQN_agent = DQN(state_dim, action_dim, cfg)
    DQN_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0

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
        count = 0
        for i in random.sample(network_list, network_batch):  # 一种网络trace下
            idx = 0
            for j in random.sample(user_list, user_batch):  # 一种用户trace下
                count += 1
                # print('trace id =' + str(i) + ' user id =' + str(j) + ' count = ' + str(count))
                state = env.reset(i, j)
                done = False
                while not done:
                    # select action
                    action = DQN_agent.choose_action(state)
                    state_tmp = state.numpy().copy()
                    # 更新环境，返回transition
                    next_state, reward, done, flag = env.step(action)
                    next_state_tmp = next_state.numpy().copy()
                    # 保存transition
                    # if flag:
                    #     reward -= 5.0
                    DQN_agent.memory.push(state_tmp, action, reward, next_state_tmp, done)
                    state = next_state  # 更新下一个状态
                    # update DQN agent
                    DQN_agent.update()
                    current_ep_reward += reward
                DQN_agent.target_net.load_state_dict(DQN_agent.policy_net.state_dict())
                idx += 1
        # if (i_episode + 1) % cfg.target_update == 0:  # update target_network
        #     DQN_agent.target_net.load_state_dict(DQN_agent.policy_net.state_dict())
        # DQN_agent.target_net.load_state_dict(DQN_agent.policy_net.state_dict())
        print_running_reward += (current_ep_reward / network_batch / user_batch)

        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                print_running_reward))
        print_running_reward = 0
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            DQN_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

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


if __name__ == '__main__':
    train()
