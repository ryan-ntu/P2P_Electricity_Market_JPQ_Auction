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
    def __init__(self, obs_dim, action_dim, hidden_1=512, hidden_2=512, trick=None):
        super(Actor, self).__init__()

        self.trick = trick or {}

        # 直接将完整观测展平后输入 MLP
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        if self.trick.get("orthogonal_init", False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.mean_layer, gain=0.01)

    def forward(self, x):
        if self.trick.get("feature_norm", False):
            x = F.layer_norm(x, x.size()[1:])

        h = F.relu(self.l1(x))
        if self.trick.get("LayerNorm", False):
            h = F.layer_norm(h, h.size()[1:])
        h = F.relu(self.l2(h))
        if self.trick.get("LayerNorm", False):
            h = F.layer_norm(h, h.size()[1:])

        mean = torch.sigmoid(self.mean_layer(h))
        log_std = self.log_std.expand_as(mean)
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
    def __init__(self, num_agents: int, obs_dim: int, hidden_1: int = 2056, hidden_2: int = 1024, trick=None):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.trick = trick or {}

        total_dim = obs_dim * num_agents
        self.l1 = nn.Linear(total_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, hidden_2 // 4)
        self.l4 = nn.Linear(hidden_2 // 4, num_agents)

        if self.trick.get("orthogonal_init", False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3)
            net_init(self.l4)

    def forward(self, s_list):
        per_agent_obs = list(s_list)
        fused = torch.cat(per_agent_obs, dim=1)

        if self.trick.get("feature_norm", False):
            fused = F.layer_norm(fused, fused.size()[1:])

        q = F.relu(self.l1(fused))
        if self.trick.get("LayerNorm", False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        if self.trick.get("LayerNorm", False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l3(q))
        if self.trick.get("LayerNorm", False):
            q = F.layer_norm(q, q.size()[1:])
        q = self.l4(q)
        return q
    
class Agent:
    def __init__(self, obs_dim, action_dim, global_obs_dim, actor_lr, critic_lr, is_continue, device, trick, use_shared_critic: bool = True):   
        
        if is_continue:
            self.actor = Actor(obs_dim, action_dim, trick=trick ).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=trick).to(device)

        # When using a centralized/shared critic, do not create per-agent critic
        self.critic = None if use_shared_critic else Critic(global_obs_dim ,trick=trick).to(device)

        # Optimizers
        if trick['adam_eps']:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            if not use_shared_critic:
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            if not use_shared_critic:
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

class MAPPO: 
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
        # Create shared centralized critic (outputs per-agent values)
        self.critic = Critic(num_agents=num_agents, obs_dim=obs_dim, trick=trick).to(device) # 构造critic
        if trick['adam_eps']:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        for i in range(num_agents):
            # Each agent only has its own actor; critic is shared
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
        obs, action, reward, next_obs, done , action_log_pi , adv_dones = self.all() # 从buffer中取出一段轨迹 这个轨迹的长度是self.horizon
        # 这里的reward 是每个状态每个agent的reward
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(self.horizon, self.num_agents) #存的是每个agent的gae
            gae = 0 # 初始化gae
            # Shared critic outputs per-agent values: (B, N)
            vs = self.critic(obs.values())           # batch_size x num_agents 获得当前obs 这里传进去的是一个dict_values
            vs_ = self.critic(next_obs.values())     # batch_size x num_agents 获得下个状态的obs

            reward = torch.cat(list(reward.values()), dim = 1) # 当前状态的reward
            done = torch.cat(list(done.values()), dim = 1)
            adv_dones = torch.cat(list(adv_dones.values()), dim = 1) # 是否被截断

            td_delta = reward + gamma * (1.0 - done) * vs_ - vs  #计算每个 状态的td误差
            
            for i in reversed(range(self.horizon)):
                gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i]) # 计算每个状态的gae
                adv[i] = gae # 保存到adv中  720*4 从这一步往后看，这个动作总体上比平均水平好多少

            adv = adv.to(self.device)  # batch_size x num_agents
            v_target = adv + vs  # batch_size x num_agents
            if self.trick['adv_norm']:  
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8)) 

            agent_ids = list(self.agents.keys())
            adv_per_agent = {
                agent_id: adv[:, idx:idx + 1]
                for idx, agent_id in enumerate(agent_ids)
            }

        # Optimize policy for K epochs
        for _ in range(K_epochs): 
            # 随机打乱样本 并 生成小批量
            shuffled_indices = np.random.permutation(self.horizon)
            indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, self.horizon, minibatch_size)] # 随机去mini batch 索引
            for index in indexes:
                # 先更新每个 actor（去中心化）
                for agent_id, agent in self.agents.items(): # 对于每个agent来说
                    agent_adv = adv_per_agent[agent_id][index]
                    if self.is_continue:
                        mean, std = agent.actor(obs[agent_id][index]) # 获得mean 和 std 
                        dist_now = Normal(mean, std) # 构造正态分布
                        dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1 策略分布的熵
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index]) # mini_batch_size x action_dim action dim = 3
                    else:
                        dist_now = Categorical(probs=agent.actor(obs[agent_id][index]))
                        dist_entropy = dist_now.entropy().reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                        action_log_pi_now = dist_now.log_prob(action[agent_id][index].reshape(-1)).reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1

                    ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi[agent_id][index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1) 看新的策略更喜欢原来的策略了还是更不喜欢原来的策略了
                    surr1 = ratios * agent_adv   # mini_batch_size x 1
                    surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * agent_adv # 使用该agent的advantage
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean() # 提供一个约束 让学习尽可能保证随机
                    agent.update_actor(actor_loss) #更新actor

                # 再更新共享 Critic（集中式）
                obs_batch = {aid: obs[aid][index] for aid in obs.keys()}
                v_s = self.critic(obs_batch.values())  # mini_batch_size x num_agents
                v_target_ = v_target[index]
                if self.trick['ValueClip']:
                    v_target_clip = torch.clamp(v_target_, v_s - clip_param, v_s + clip_param)
                    if self.trick['huber_loss']:
                        critic_loss_clip = huber_loss(v_target_clip - v_s, huber_delta).mean()
                        critic_loss_original = huber_loss(v_target_ - v_s, huber_delta).mean()
                    else:
                        critic_loss_clip = F.mse_loss(v_target_clip, v_s)
                        critic_loss_original = F.mse_loss(v_target_, v_s)
                    critic_loss = torch.max(critic_loss_original, critic_loss_clip)
                else:
                    if self.trick['huber_loss']: # huber loss是为了稳定误差的
                        critic_loss = huber_loss(v_target_ - v_s, huber_delta).mean()
                    else:
                        critic_loss = F.mse_loss(v_target_, v_s) # 用mse loss来更新critical 
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        

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
        save_data = {name: agent.actor.state_dict() for name, agent in self.agents.items()}
        save_data['critic'] = self.critic.state_dict()
        
        torch.save(
            save_data,
            os.path.join(model_dir, f'MAPPO_{name}_pay.pth')
        )
        
    ## 加载模型
    @staticmethod 
    def load(num_agents, obs_dim, action_dim, is_continue, model_dir, trick=None, horizon=1000, device='cpu'):
        # 使用指定的设备来初始化模型
        policy = MAPPO(num_agents, obs_dim, action_dim, is_continue = is_continue, actor_lr = 0, critic_lr = 0, horizon = horizon, device = device, trick=trick)
        data = torch.load(model_dir, map_location=torch.device(device), weights_only=False)
        
        # 加载每个agent的actor模型
        for agent_id, agent in policy.agents.items():
            if agent_id in data:
                agent.actor.load_state_dict(data[agent_id])
        
        # 加载共享的critic模型（如果存在）
        if 'critic' in data:
            policy.critic.load_state_dict(data['critic'])
        
        return policy


