import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal,Categorical
import os

from algorithm.buffer import PrioritizedReplayBuffer, Buffer_for_PPO
import numpy as np


def net_init(m,gain=None,use_relu = True):
    '''网络初始化
    m:layer = nn.Linear(3, 2) # 按ctrl点击Linear 可知默认初始化为 nn.init.kaiming_uniform_(self.weight) ,nn.init.uniform_(self.bias) 此初始化的推荐的非线性激活函数方式为relu,和leaky_relu)
    参考2：Orthogonal Initialization trick:（原代码也是如此）
    critic: gain :nn.init.calculate_gain(['tanh', 'relu'][use_ReLU]) ; weight: nn.init.orthogonal_(self.weight, gain) ; bias: nn.init.constant_(self.bias, 0)
    actor: 其余层和critic一样，最后输出层gain = 0.01
    参考：
    1.https://zhuanlan.zhihu.com/p/210137182， -> orthogonal_ 优于 xavier_uniform_
    2.https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/  -> kaiming_uniform_ 替代 xavier_uniform_
    代码参考 原论文代码：https://github.com/marlbenchmark/on-policy/
    '''
    use_orthogonal = True # -> 1
    use_relu = use_relu

    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_]
    activate_fuction = ['tanh','relu', 'leaky_relu']  # relu 和 leaky_relu 的gain值一样
    gain = gain if gain is not None else  nn.init.calculate_gain(activate_fuction[use_relu]) # 根据的激活函数设置
    
    init_method[use_orthogonal](m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=512, hidden_2=512,trick = None):
        super(Actor, self).__init__()

        self.trick = trick or {}

        assert obs_dim >= 33, "obs_dim must be at least 33 for layout [3 scalars] + [7×4 window] + [time]"

        # LSTM encoder for window (t-1, t, t+1, t+2, t+3, t+4, t+5): sequence length=7, feature=4
        self.ts_lstm = nn.LSTM(input_size=4, hidden_size=hidden_1 // 4, num_layers=1, batch_first=True)
        self.ts_out = nn.Linear(hidden_1 // 4, hidden_2 // 64)

        self.scalar_proj = nn.Linear(3, hidden_1 // 4)
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_1 // 64),
            nn.ReLU(inplace=True),
        )

        fused_dim = (hidden_1 // 4) + (hidden_2 // 64) + (hidden_1 // 64)

        self.l1 = nn.Linear(fused_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # 与PPO.py的方法一致：对角高斯函数
        #self.log_std_layer = nn.Linear(hidden_2, action_dim) # 式2

        self.trick = trick
        # 使用 orthogonal_init
        if self.trick.get('orthogonal_init', False):
            # LSTM weights: orthogonal for hidden, zeros for biases
            for name, param in self.ts_lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
            net_init(self.ts_out)
            net_init(self.scalar_proj)
            for layer in self.time_proj:
                if isinstance(layer, nn.Linear):
                    net_init(layer)
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.mean_layer, gain=0.01)  

    def forward(self, x):

        """
        x: (B, obs_dim)
        returns: mean (B, action_dim), std (B, action_dim)
        """
        x_scalar = x[:, :3]                  # (B,3)
        x_window = x[:, 3:31].view(-1, 7, 4) # (B,7,4)  window: t-1, t, t+1, t+2, t+3, t+4, t+5
        x_time = x[:, 31:33]                 # (B,2)

        # scalar embedding
        s = F.relu(self.scalar_proj(x_scalar))  # (B, hidden_1//4)

        # time-series window via LSTM
        if self.trick.get('feature_norm', False):
            # optional layer norm on raw sequence (normalize per feature across time)
            # apply layer_norm per time-step feature dim = 3
            x_window = F.layer_norm(x_window, x_window.size()[1:])  # normalize seq length & feature jointly if desired
        _, (h_n, _) = self.ts_lstm(x_window)  # h_n: (1, B, hidden_1//4)
        ts_feat = h_n[-1]                     # (B, hidden_1//4)
        ts_emb = F.relu(self.ts_out(ts_feat))  # (B, hidden_2//64)
        t = self.time_proj(x_time)                # (B, hidden_1//4)

        # fuse features
        h = torch.cat([s, ts_emb, t], dim=1)  # (B, fused_dim)
        if self.trick.get('feature_norm', False):
            h = F.layer_norm(h, h.size()[1:])

        h = F.relu(self.l1(h))
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

    
class Actor_discrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128,trick = None):
        super(Actor_discrete, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

        self.trick = trick
        # 使用 orthogonal_init
        if trick['orthogonal_init']:
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3, gain=0.01) 

    def forward(self, obs ):
        if self.trick['feature_norm']:
            obs = F.layer_norm(obs, obs.size()[1:])
        x = F.relu(self.l1(obs))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick['LayerNorm']:
            x = F.layer_norm(x, x.size()[1:])
        a_prob = torch.softmax(self.l3(x), dim=1)
        return a_prob
        
class Critic(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int, hidden_1: int = 2056, hidden_2: int = 1024, trick = None):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim

        # LSTM over multi-agent time windows: sequence length=7, input feature per step = 4*num_agents
        self.ts_lstm = nn.LSTM(input_size=4 * num_agents, hidden_size=hidden_1 // 8, num_layers=1, batch_first=True)
        self.ts_out = nn.Linear(hidden_1 // 8, hidden_2 // 64)

        self.time_embed_dim = max(hidden_1 // 256, 1)
        self.time_proj = nn.Sequential(
            nn.Linear(2, self.time_embed_dim),
            nn.ReLU(inplace=True),
        )

        # After LSTM, fuse with all agents' scalar(3) + processed time features
        fused_dim = (hidden_2 // 64) + (3 + self.time_embed_dim) * num_agents

        self.l1 = nn.Linear(fused_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, hidden_2 // 4)
        # Output per-agent value (centralized but multi-head)
        self.l4 = nn.Linear(hidden_2 // 4, 1)

        self.trick = trick or {}
        if self.trick.get('orthogonal_init', False):
            for name, param in self.ts_lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
            net_init(self.ts_out)
            for layer in self.time_proj:
                if isinstance(layer, nn.Linear):
                    net_init(layer)
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3)
            net_init(self.l4)

    def forward(self, s_list):
        # s_list: iterable of per-agent obs tensors, each (B, obs_dim)
        per_agent_obs = list(s_list) # 将dict_values转换为list
        B = per_agent_obs[0].shape[0] # 获取观batch_size
        N = self.num_agents

        # Extract per-agent windows (B,7,4) and stack along feature dim -> (B,7,4*N)
        windows = []
        scalars_time = []
        for a_obs in per_agent_obs: # 遍历每个agent的观测
            a_scalar = a_obs[:, :3]      # (B,3)
            a_window = a_obs[:, 3:31]    # (B,28)
            a_time = a_obs[:, 31:33]     # (B,2) 获取观测的每个部分的信息
            a_window = a_window.view(B, 7, 4)
            windows.append(a_window)
            time_emb = self.time_proj(a_time)
            scalars_time.append(torch.cat([a_scalar, time_emb], dim=1))  # (B,3+time_emb_dim)
        # concat features across agents for each time step
        seq = torch.cat(windows, dim=2)  # (B,7,4*N) 将agent的所有观测拼接起来
        if self.trick.get('feature_norm', False):
            seq = F.layer_norm(seq, seq.size()[1:])
        _, (h_n, _) = self.ts_lstm(seq)
        ts_feat = h_n[-1]                  # (B, hidden_1//4)
        ts_emb = F.relu(self.ts_out(ts_feat))  # (B, hidden_2//64)

        scalars_time_all = torch.cat(scalars_time, dim=1)  # (B, 5*N)
        fused = torch.cat([ts_emb, scalars_time_all], dim=1)
        if self.trick.get('feature_norm', False):
            fused = F.layer_norm(fused, fused.size()[1:])

        q = F.relu(self.l1(fused))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l3(q))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = self.l4(q)  # (B, N)
        return q
    
class Agent:
    def __init__(self, num_agents, obs_dim, action_dim, global_obs_dim, actor_lr, critic_lr, is_continue, device, trick):   
        
        if is_continue:
            self.actor = Actor(obs_dim, action_dim, trick=trick ).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=trick).to(device)

        # When using a centralized/shared critic, do not create per-agent critic
        self.critic = Critic(num_agents, obs_dim ,trick=trick).to(device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr= actor_lr)

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

## 第二部分：定义DQN算法类
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

class MAPPO_i: 
    def __init__(self, num_agents, obs_dim, action_dim, is_continue, actor_lr, critic_lr, horizon, device, trick = None):        
        # Ensure 'trick' is a complete dict even when None or missing keys
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
            # Each agent only has its own actor; critic is shared
            self.agents[f'agent_{i}'] = Agent(
                num_agents,obs_dim, action_dim, global_obs_dim, actor_lr, critic_lr,
                is_continue, device, trick)
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
    
    ## buffer 相关
    def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id], action_log_pi[agent_id] , adv_dones[agent_id])

    def all(self):
        obs = {}
        action = {}
        reward = {}
        next_obs = {}
        done = {}
        action_log_pi = {}
        adv_dones = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id], action_log_pi[agent_id], adv_dones[agent_id] = buffer.all()
        return obs, action, reward, next_obs, done, action_log_pi, adv_dones

    ## PPO算法相关
    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient, huber_delta = None):
        # 多智能体特有-- 集中式训练critic:要用到所有智能体next状态和动作
        obs, action, reward, next_obs, done , action_log_pi , adv_dones = self.all()
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(self.horizon, self.num_agents)
            gae = 0
            vs = []
            vs_ = []
            for agent_id  in self.buffers.keys():
                vs.append(self.agents[agent_id].critic(obs.values()))  # batch_size x 1
                vs_.append(self.agents[agent_id].critic(next_obs.values()))
            
            vs = torch.cat(vs, dim = 1) # batch_size x 3
            vs_ = torch.cat(vs_, dim = 1) # batch_size x 3

            reward = torch.cat(list(reward.values()), dim = 1) # 
            done = torch.cat(list(done.values()), dim = 1)
            adv_dones = torch.cat(list(adv_dones.values()), dim = 1)

            td_delta = reward + gamma * (1.0 - done) * vs_ - vs  #这里可能使用全局的reward
            
            for i in reversed(range(self.horizon)):
                gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i])
                adv[i] = gae

            adv = adv.to(self.device)  # batch_size x 3
            v_target = adv + vs  # batch_size x 3
            if self.trick['adv_norm']:  
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8)) 

        agent_ids = list(self.agents.keys())
        adv_per_agent = {
            agent_id: adv[:, idx:idx + 1]
            for idx, agent_id in enumerate(agent_ids)
        }

        for agent_id, agent in self.agents.items():
            # Optimize policy for K epochs:
            for _ in range(K_epochs): 
                # 随机打乱样本 并 生成小批量
                shuffled_indices = np.random.permutation(self.horizon)
                indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)]
                for index in indexes:
                    # 先更新actor
                    if self.is_continue:
                        mean, std = self.agents[agent_id].actor(obs[agent_id][index])
                        dist_now = Normal(mean, std)
                        dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index]) # mini_batch_size x action_dim
                    else:
                        dist_now = Categorical(probs=self.agents[agent_id].actor(obs[agent_id][index]))
                        dist_entropy = dist_now.entropy().reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index].reshape(-1)).reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1

                    agent_adv = adv_per_agent[agent_id][index]
                    ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi[agent_id][index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1)
                    surr1 = ratios * agent_adv
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * agent_adv 
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean()
                    agent.update_actor(actor_loss)

                    # 再更新critic
                    obs_ = {agent_id: obs[agent_id][index] for agent_id in obs.keys()}

                    v_s = self.agents[agent_id].critic(obs_.values()) # mini_batch_size x 1
                    v_s = v_s.repeat(1,self.num_agents) # mini_batch_size x 3

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
                    agent.update_critic(critic_loss)
        

        ## 清空buffer
        for buffer in self.buffers.values():
            buffer.clear()
    
    def lr_decay(self,episode_num,max_episodes):
        lr_a_now = self.actor_lr * (1 - episode_num / max_episodes)
        lr_c_now = self.critic_lr * (1 - episode_num / max_episodes)
        for agent in self.agents.values():
            for p in agent.actor_optimizer.param_groups:
                p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now
 

    def save(self, model_dir,name):
        # 保存所有agent的actor模型和共享的critic模型
        save_data = {}
        for agent_name, agent in self.agents.items():
            save_data[f'{agent_name}/actor'] = agent.actor.state_dict()
            save_data[f'{agent_name}/critic'] = agent.critic.state_dict()
        torch.save(save_data, os.path.join(model_dir, f'MAPPO_i_{name}.pth'))
        
    ## 加载模型
    @staticmethod 
    def load(num_agents, obs_dim, action_dim, is_continue, model_dir, trick=None, horizon=1000, device='cpu'):
        # 使用指定的设备来初始化模型
        policy = MAPPO_i(num_agents, obs_dim, action_dim, is_continue = is_continue, actor_lr = 0, critic_lr = 0, horizon = horizon, device = device, trick=trick)
        data = torch.load(model_dir, map_location=torch.device(device), weights_only=False)
        
        # 加载每个agent的actor模型
        for agent_id, agent in policy.agents.items():
            if agent_id in data:
                agent.actor.load_state_dict(data[agent_id])
        
        # 加载共享的critic模型（如果存在）
        if 'critic' in data:
            policy.critic.load_state_dict(data['critic'])
        
        return policy


