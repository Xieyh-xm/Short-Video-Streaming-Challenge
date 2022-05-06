import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 15),
                nn.Softmax(dim=-1)
            )
            self.actor.to(device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.critic.to(device)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self, input):
        raise NotImplementedError

    def act(self, state):
        state_numpy = state.numpy()  # CPU
        ifsleep = False
        chunk_last = state_numpy[0, 20:25]
        buffer = state_numpy[0, 15:20]
        # print_debug('chunk_last = ' + str(chunk_last))
        mask = np.zeros(15)
        for i in range(5):
            for j in range(3):
                if chunk_last[i] != 0.0:
                    mask[i * 3 + j] = 1
        if buffer[0] <= 1. and chunk_last[0] != 0:
            for i in range(3, 15):
                mask[i] = 0
        # print_debug('mask = ' + str(mask))
        mask = torch.tensor(mask).to(device)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # for i in range(235):
            #     state[0,i]=1
            action_probs = self.actor(state)

            action_probs = action_probs * mask
            # print_debug('遮蔽后action_probs =' + str(action_probs))
            if torch.sum(action_probs) != 0:
                dist = Categorical(action_probs)
            else:
                ifsleep = True
                action_probs[0][0] = 1.0
                dist = Categorical(action_probs)
                # print_debug(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), ifsleep

    def evaluate(self, state, action):
        state_numpy = state.numpy()  # CPU
        chunk_last = state_numpy[:, 20:25]
        buffer = state_numpy[:, 15:20]
        mask = np.zeros((state_numpy.shape[0], 15))
        for k in range(state_numpy.shape[0]):
            for i in range(5):
                for j in range(3):
                    if chunk_last[k, i] != 0.0:
                        mask[k, i * 3 + j] = 1
        for k in range(state_numpy.shape[0]):
            if buffer[k, 0] <= 1. and chunk_last[k, 0] != 0:
                for i in range(3, 15):
                    mask[k, i] = 0
        mask = torch.tensor(mask).to(device)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            action_probs = action_probs * mask

            for i in range(state_numpy.shape[0]):
                if torch.sum(action_probs[i]) <= 0:
                    action_probs[i][0] = 1.0
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
            self.set_action_std(self.action_std)

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, sleep_flag = self.policy_old.act(state)
            state_n = torch.zeros([1, 30])

            state_n[0, :] = state[0, :]
            self.buffer.states.append(state_n)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            state_numpy = state.numpy()
            history_throughput = state_numpy[0, 0:5]
            ave_throughput = history_throughput.sum() / 5.
            chunk_last = state_numpy[0, 20:25]
            buffer = state_numpy[0, 15:20]
            action_trans = np.zeros(3)
            action_trans[0] = action // 3  # 视频id
            action_trans[1] = action % 3  # 比特率
            action_trans[2] = 0.0
            throughput4next = (action_trans[1]*0.6+1) * state_numpy[0, int(5 + action_trans[0])]
            if chunk_last[int(action_trans[0])] == 0:
                action_trans[2] = 100
            # 总共buffer大于6秒，正在观看视频buffer大于3秒，下个视频下载小于1秒，休眠1秒减下个下载视频所需要的时间
            if buffer.sum() >= 6 and buffer[0] >= 3 and throughput4next < ave_throughput:
                action_trans[2] = int((1 - throughput4next / ave_throughput) * 1000.)
            # if sleep_flag:
            #     action_trans[2] = 500
            return action_trans


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path))
        self.policy.load_state_dict(torch.load(checkpoint_path))
