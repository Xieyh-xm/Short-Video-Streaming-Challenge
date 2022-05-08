import os
import glob
import time
from datetime import datetime
import random
import torch
import numpy as np
import time

# from RL_env import RLEnv
# from RL_env_new import RLEnv
# from RL_env_new_sleep_pro import RLEnv

# from PPO import PPO
# from PPO_new import PPO

# 加上sleep flag
from PPO.PPO_sleep import PPO
from RL_env.RL_env_sleep import RLEnv


################################### Training ###################################
def train():
    print("============================================================================================")
    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)

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

    env = RLEnv()

    # state space dimension
    state_dim = env.observation_space.n

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
    env_name = '20220508'

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    # print("current logging run number for " + env_name + " : ", run_num)
    # print("logging at : " + log_f_name)
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
    ppo_agent.load(directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, 100))
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    time_step = 100
    i_episode = 0
    network_batch = 10
    network_dict_size = 1421
    user_batch = 15
    user_dict_size = 1000
    save_model_freq = 50  # save model frequency (in num timesteps)
    # training loop
    network_list = range(network_dict_size)
    user_list = range(user_dict_size)
    # state:len(state)=235
    # state[0:10]=过去10个chunk吞吐量
    # state[10:20]=过去10个chunk下载时间（delay）
    # state[20:170] = 五个视频三个质量未来10个chunk的videosize
    # state[170:220]= 五个视频的条件retent-rate
    # state[220:225]= 五个视频的buffer
    # state[225:230]= 五个视频的剩余chunk数
    # state[230:235]= 五个视频的上个质量等级
    # max_training_timesteps = 300
    while time_step <= max_training_timesteps:
        current_ep_reward = 0
        time_step += 1
        count = 0
        ticks = time.time()
        random.seed(ticks)
        for i in random.sample(network_list, network_batch):
            for j in random.sample(user_list, user_batch):
                count += 1
                state = env.reset(i, j)
                # state = env.reset(0, 0)
                # print('trace id =' + str(i) + ' user id =' + str(j) + ' count = ' + str(count))
                done = False
                cur_reward = 0
                while not done:
                    # select action with policy
                    action = ppo_agent.select_action(state)
                    state, reward, done = env.step(action)
                    # saving reward and is_terminals
                    ppo_agent.buffer.rewards.append(reward)
                    ppo_agent.buffer.is_terminals.append(done)
                    current_ep_reward += reward
                    cur_reward += reward
        # update PPO agent
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                current_ep_reward / network_batch / user_batch))
        ppo_agent.update()

        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, time_step)
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        i_episode += 1

    # log_f.close()
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
