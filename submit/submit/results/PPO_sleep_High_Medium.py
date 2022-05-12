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
                nn.Linear(64, 16),
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

        # throughput取5个
        # buffer = state_numpy[0, 15:20]
        # chunk_last = state_numpy[0, 20:25]

        # throughput取10个
        buffer = state_numpy[0, 20:25]
        chunk_last = state_numpy[0, 25:30]

        mask = np.zeros(16)
        mask[15] = 1  # 允许sleep
        # 如果没有可下载的块，则不再被允许下载
        for i in range(5):
            for j in range(3):
                if chunk_last[i] != 0.0:
                    mask[i * 3 + j] = 1
        if buffer[0] <= 2. and chunk_last[0] != 0:  # 强制下载第一个视频
            mask[15] = 0  # 不允许sleep
            for i in range(3, 15):
                mask[i] = 0
            # if buffer[0] >= 1:
            #     for i in range(3, 15):
            #         mask[i] = 0
            # else:
            #     for i in range(1, 15):
            #         mask[i] = 0

        mask = torch.tensor(mask).to(device)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            action_probs = action_probs * mask
            if torch.sum(action_probs) != 0:
                dist = Categorical(action_probs)
        # print(action_probs)
        # action = dist.sample()
        action = action_probs.argmax()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), ifsleep

    def evaluate(self, state, action):
        state_numpy = state.numpy()  # CPU
        # throughput取5个
        # buffer = state_numpy[:, 15:20]
        # chunk_last = state_numpy[:, 20:25]

        # throughput取10个
        buffer = state_numpy[:, 20:25]
        chunk_last = state_numpy[:, 25:30]

        mask = np.zeros((state_numpy.shape[0], 16))
        for k in range(state_numpy.shape[0]):
            mask[k, 15] = 1
            for i in range(5):
                for j in range(3):
                    if chunk_last[k, i] != 0.0:
                        mask[k, i * 3 + j] = 1
        for k in range(state_numpy.shape[0]):
            if buffer[k, 0] <= 2 and chunk_last[k, 0] != 0:
                mask[k, 15] = 0
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
            # state_n = torch.zeros([1, 30])
            state_n = torch.zeros([1, 35])
            state_n[0, :] = state[0, :]
            self.buffer.states.append(state_n)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            action_trans = [0 for i in range(3)]
            action_trans[0] = int(action / 3)  # 视频id
            action_trans[1] = action % 3  # 比特率
            if action == 15:  # sleep_flag
                action_trans[2] = True
                action_trans[0] = 0
                action_trans[1] = 0
            else:
                action_trans[2] = False
            return action_trans

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        # print(old_states[:,225:230])
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path))
        self.policy.load_state_dict(torch.load(checkpoint_path))
