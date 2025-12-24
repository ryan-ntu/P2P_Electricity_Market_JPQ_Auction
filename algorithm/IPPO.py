from tokenize import Name
from .MAPPO import Actor_discrete, Critic, huber_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal,Categorical
import os

from algorithm.buffer import PrioritizedReplayBuffer, Buffer_for_PPO
import numpy as np

def net_init(m,gain=None,use_relu = True):
    use_orthogonal = True # -> 1
    use_relu = use_relu

    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_]
    activate_fuction = ['tanh','relu', 'leaky_relu']  # relu 和 leaky_relu 的gain值一样
    gain = gain if gain is not None else  nn.init.calculate_gain(activate_fuction[use_relu]) # 根据的激活函数设置
    
    init_method[use_orthogonal](m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=512, hidden_2=512, trick=None):
        super(Actor, self).__init__()

        self.trick = trick or {}

        # 直接将整个obs flatten后输入到MLP
        # 输入维度就是obs_dim
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # 对角高斯

        self.trick = trick or {}

        # orthogonal_init
        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.mean_layer, gain=0.01)

    def forward(self, x):
        """
        x: (B, obs_dim)
        returns: mean (B, action_dim), std (B, action_dim)
        """
        # 可选的feature_norm
        if self.trick.get('feature_norm', False):
            x = F.layer_norm(x, x.size()[1:])

        # 直接将flatten的obs输入到MLP
        h = F.relu(self.l1(x))
        if self.trick.get('LayerNorm', False):
            h = F.layer_norm(h, h.size()[1:])
        h = F.relu(self.l2(h))
        if self.trick.get('LayerNorm', False):
            h = F.layer_norm(h, h.size()[1:])

        mean = torch.sigmoid(self.mean_layer(h))  # 输出在 [0,1]
        log_std = self.log_std.expand_as(mean)    # (B, action_dim)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_1: int = 2056, hidden_2: int = 1024, trick=None):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.trick = trick or {}

        # 直接将整个obs flatten后输入到MLP
        # 输入维度就是obs_dim
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, hidden_2 // 4)
        # 输出单个值 V(s)
        self.l4 = nn.Linear(hidden_2 // 4, 1)

        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3)
            net_init(self.l4)

    def forward(self, x):
        # x: (B, obs_dim) 直接将整个obs输入到MLP
        # 可选的feature_norm
        if self.trick.get('feature_norm', False):
            x = F.layer_norm(x, x.size()[1:])

        q = F.relu(self.l1(x))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l3(q))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])

        v = self.l4(q)  # (B,1)
        return v
    


class Agent:
    def __init__(self, obs_dim, action_dim, global_obs_dim, actor_lr, critic_lr, is_continue, device, trick, use_shared_critic: bool = True):   
        
        if is_continue:
            self.actor = Actor(obs_dim, action_dim, trick=trick ).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=trick).to(device)

        # 这里的agent需要给每个agent构造一个critic
        self.critic = Critic(obs_dim ,trick=trick).to(device)

        # Optimizers
        if trick['adam_eps']:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        # Only valid when using per-agent critic (not used in shared-critic mode)
        if hasattr(self, 'critic_optimizer') and self.critic is not None:
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

    def update_ac(self, loss):
        self.ac_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10)
        self.ac_optimizer.step()


class IPPO():
    def __init__(self, num_agents, obs_dim, action_dim, is_continue, actor_lr, critic_lr, horizon, device, trick = None):        
        default_trick = {
            'adv_norm': False,
            'ObsNorm': False,
            'reward_norm': False,
            'reward_scaling': False,
            'orthogonal_init': False,
            'adam_eps': False,
            'lr_decay': False,
            'ValueClip': False,
            'huber_loss': False,
            'LayerNorm': False,
            'feature_norm': False,
        }
        if trick is None:
            trick = default_trick.copy()
        else:
            # Fill missing keys with default values
            for k, v in default_trick.items():
                trick.setdefault(k, v)

        self.agents  = {}
        self.buffers = {}
        global_obs_dim = obs_dim * num_agents 
        for i in range(num_agents):
            self.agents[f'agent_{i}'] = Agent(
                obs_dim, action_dim, global_obs_dim, actor_lr, critic_lr,
                is_continue, device, trick, use_shared_critic=True
            ) # 构造多个agent 
            self.buffers[f'agent_{i}'] = Buffer_for_PPO(horizon, obs_dim, act_dim = action_dim if is_continue else 1, device = device) # 为每个agent构造一个buffer

        self.device = device
        self.is_continue = is_continue
        print('actor_type:continue') if self.is_continue else print('actor_type:discrete')

        self.horizon = int(horizon)

        self.trick = trick
        self.num_agents = len(self.agents) 

        if self.trick['lr_decay']:
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr


    def select_action(self, obs):
        actions = {}
        action_log_pis = {}
        for agent_id, obs in obs.items(): # 遍历每个obs 牛了逼了 这里没报错
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device) # 将观测转变成tensor
            if self.is_continue: 
                mean, std = self.agents[agent_id].actor(obs) # 获取每个agent的均值和方差
                dist = Normal(mean, std) # 构造正态分布
                action = dist.sample() # sample一个
                action_log_pi = dist.log_prob(action) # 1xaction_dim 获取每个动作的对数概率 
            else:
                dist = Categorical(probs=self.agents[agent_id].actor(obs)) # 构造分类分布
                action = dist.sample()
                action_log_pi = dist.log_prob(action)
            # to 真实值
            actions[agent_id] = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim 记录每个agent 的action和对应的log prob
            action_log_pis[agent_id] = action_log_pi.detach().cpu().numpy().squeeze(0)

        return actions , action_log_pis
    
    def evaluate_action(self,obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: 
                mean, _ = self.agents[agent_id].actor(obs)
                action = mean.detach().cpu().numpy().squeeze(0)
            else:
                a_prob = self.agents[agent_id].actor(obs).detach().cpu().numpy().squeeze(0)
                action = np.argmax(a_prob)

            actions[agent_id] = action
        return actions
    

    def evaluate_action(self,obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: 
                mean, _ = self.agents[agent_id].actor(obs)
                action = mean.detach().cpu().numpy().squeeze(0)
            else:
                a_prob = self.agents[agent_id].actor(obs).detach().cpu().numpy().squeeze(0)
                action = np.argmax(a_prob)

            actions[agent_id] = action
        return actions

    def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id], action_log_pi[agent_id] , adv_dones[agent_id])


    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient, huber_delta = None):
        # 多智能体特有-- 集中式训练critic:要用到所有智能体next状态和动作
        for agent_id in self.agents.keys():
            obs, action, reward, next_obs, done , action_log_pi , adv_dones = self.buffers[agent_id].all()
            # 计算GAE
            with torch.no_grad():  # adv and v_target have no gradient
                adv = np.zeros(self.horizon)
                gae = 0
                vs = self.agents[agent_id].critic(obs) # obs 的大小是horizon(buffer size) * obs_dim 一股脑全丢进给critic了
                vs_ = self.agents[agent_id].critic(next_obs) # buffer size * 1
                td_delta = reward + gamma * (1.0 - done) * vs_ - vs # 计算td误差 
                td_delta = td_delta.reshape(-1).cpu().detach().numpy()
                adv_dones = adv_dones.reshape(-1).cpu().detach().numpy()
                for i in reversed(range(self.horizon)):
                    gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i])
                    adv[i] = gae # list: buffer size
                adv = torch.as_tensor(adv,dtype=torch.float32).reshape(-1, 1).to(self.device) ## cuda buffer size * 1
                v_target = adv + vs  # critical target
                if self.trick['adv_norm']:  
                    adv = ((adv - adv.mean()) / (adv.std() + 1e-8)) 
                  

            # Optimize policy for K epochs:
            for _ in range(K_epochs): 
                # 随机打乱样本 并 生成小批量
                shuffled_indices = np.random.permutation(self.horizon) # 给当前的agent 生成小批量样本
                indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)]
                for index in indexes:
                    # 先更新actor
                    if self.is_continue:
                        mean, std = self.agents[agent_id].actor(obs[index])
                        dist_now = Normal(mean, std)
                        dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[index]) # mini_batch_size x action_dim
                    else:
                        dist_now = Categorical(probs=self.agents[agent_id].actor(obs[index]))
                        dist_entropy = dist_now.entropy().reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[index].reshape(-1)).reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1

                    ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi[index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1)
                    surr1 = ratios * adv[index]   # mini_batch_size x 1
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv[index]  
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean()
                    self.agents[agent_id].update_actor(actor_loss)

                    # 再更新每个agent对应的critic
                    #obs_ = {agent_id: obs[index] for agent_id in self.agents.keys()}
                    #obs_ = obs[index]
                    v_s = self.agents[agent_id].critic(obs[index])
                    v_target_ = v_target[index]
                    if self.trick['ValueClip']:
                        ''' 参考原mappo代码,原代码存储了return和value值,故实现上和如下有些许差异'''
                        v_target_clip = torch.clamp(v_target_, v_s - clip_param, v_s + clip_param)
                        if self.trick['huber_loss']:
                            critic_loss_clip = huber_loss(v_target_clip-v_s,huber_delta).mean()
                            critic_loss_original = huber_loss(v_target_-v_s,huber_delta).mean()
                        else:
                            critic_loss_clip = F.mse_loss(v_target_clip, v_s)
                            critic_loss_original = F.mse_loss(v_target_, v_s)
                        critic_loss = torch.max(critic_loss_original,critic_loss_clip)
                    else:
                        if self.trick['huber_loss']:
                            critic_loss = huber_loss(v_target_-v_s,huber_delta).mean()
                        else:
                            critic_loss = F.mse_loss(v_target_, v_s)
                    self.agents[agent_id].update_critic(critic_loss)
        

        ## 清空buffer
        for buffer in self.buffers.values():
            buffer.clear() # 清空所有agent的buffer
    

    def lr_decay(self,episode_num,max_episodes):
        lr_a_now = self.actor_lr * (1 - episode_num / max_episodes)
        lr_c_now = self.critic_lr * (1 - episode_num / max_episodes)
        for agent in self.agents.values():
            for p in agent.actor_optimizer.param_groups:
                p['lr'] = lr_a_now
            for p in agent.critic_optimizer.param_groups:
                p['lr'] = lr_c_now
 

    def save(self, model_dir, name):
        save_data = {}
        for agent_name, agent in self.agents.items():
            save_data[f'{agent_name}/actor'] = agent.actor.state_dict()
            save_data[f'{agent_name}/critic'] = agent.critic.state_dict()
        torch.save(save_data, os.path.join(model_dir, f'IPPO_{name}.pth'))

    ## 加载模型
    @staticmethod
    def load(dim_info, is_continue, model_dir, trick=None):
        policy = IPPO(dim_info, is_continue=is_continue, actor_lr=0, critic_lr=0, horizon=0, device='cpu', trick=trick)
        data = torch.load(os.path.join(model_dir, 'IPPO.pth'))
        for agent_id, agent in policy.agents.items():
            if f'{agent_id}/actor' in data:
                agent.actor.load_state_dict(data[f'{agent_id}/actor'])
            if f'{agent_id}/critic' in data:
                agent.critic.load_state_dict(data[f'{agent_id}/critic'])
        return policy