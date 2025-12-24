import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from copy import deepcopy
from algorithm.buffer import Buffer_for_DDPG
import numpy as np
import os


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))  # 输出在 [0,1] 范围，对应环境的动作空间
        return x


class Critic(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, hidden_1=128, hidden_2=128):
        super(Critic, self).__init__()
        # 全局观测和动作的维度：所有智能体的观测 + 所有智能体的动作
        global_obs_act_dim = obs_dim * num_agents + action_dim * num_agents
        
        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, s, a):
        """
        传入全局观测和动作
        s: iterable of per-agent obs tensors, each (B, obs_dim)
        a: iterable of per-agent action tensors, each (B, action_dim)
        returns: Q value (B, 1)
        """
        # 将观测和动作拼接
        sa = torch.cat(list(s) + list(a), dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class Agent:
    def __init__(self, obs_dim, action_dim, num_agents, actor_lr, critic_lr, device, trick=None):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(num_agents, obs_dim, action_dim).to(device)
        
        if trick and trick.get('adam_eps', False):
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MADDPG:
    def __init__(self, num_agents, obs_dim, action_dim, is_continue, actor_lr, critic_lr, horizon, device, trick=None):
        # Ensure 'trick' is a complete dict even when None or missing keys
        default_trick = {
            'orthogonal_init': False,
            'adam_eps': False,
            'LayerNorm': False,
            'feature_norm': False,
        }
        if trick is None:
            trick = default_trick.copy()
        else:
            # Fill missing keys with default values
            for k, v in default_trick.items():
                trick.setdefault(k, v)
        
        self.agents = {}
        self.buffers = {}
        
        # 为每个智能体创建 Actor、Critic 和 Buffer
        for i in range(num_agents):
            agent_id = f'agent_{i}'
            self.agents[agent_id] = Agent(obs_dim, action_dim, num_agents, actor_lr, critic_lr, device, trick)
            # 使用 Buffer_for_DDPG，buffer_size 使用 horizon 参数
            self.buffers[agent_id] = Buffer_for_DDPG(horizon, obs_dim, action_dim, device)
        
        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0]  # sample 用
        self.regular = False  # 与DDPG中使用的weight_decay原理一致
        self.trick = trick
        
        print('MADDPG actor_type: continue') if self.is_continue else print('MADDPG actor_type: discrete')

    def select_action(self, obs):
        actions = {}
        log_probs = {}  # MADDPG 不需要 log_probs，但为了接口兼容性返回 None
        for agent_id, obs_val in obs.items():
            obs_tensor = torch.as_tensor(obs_val, dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue:
                action = self.agents[agent_id].actor(obs_tensor)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)  # 1xaction_dim -> action_dim
                log_probs[agent_id] = None  # MADDPG 不需要 log_probs
            else:
                raise NotImplementedError("Discrete action space not implemented")
        return actions, log_probs

    def evaluate_action(self, obs):
        actions = {}
        for agent_id, obs_val in obs.items():
            obs_tensor = torch.as_tensor(obs_val, dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue:
                action = self.agents[agent_id].actor(obs_tensor)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)  # 1xaction_dim -> action_dim
        return actions

    def add(self, obs, action, reward, next_obs, done, log_probs=None, adv_done=None):
        """
        添加经验到buffer
        log_probs 和 adv_done 参数为了与主训练文件接口兼容，但 MADDPG 不使用
        """
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

    def sample(self, batch_size):
        total_size = len(self.buffers[self.agent_x])
        if total_size == 0:
            return None, None, None, None, None, None
        if total_size < batch_size:
            batch_size = total_size
        indices = np.random.choice(total_size, batch_size, replace=False)
        
        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        next_action = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)
            # 使用 target actor 计算 next_action
            with torch.no_grad():
                next_action[agent_id] = self.agents[agent_id].actor_target(next_obs[agent_id])
        
        return obs, action, reward, next_obs, done, next_action  # 包含所有智能体的数据

    def learn(self, batch_size, gamma, tau):
        """
        MADDPG 学习更新
        论文中提出两种方法更新actor，这里选择方式0实现：
        0. actor_loss = -critic(x, actor(obs), other_act).mean() 知道其他agent的策略来更新
        1. actor_loss = -(log(actor(obs)) + lmbda * H(actor_dist)) 知道其他智能体的obs但不知道策略来更新
        """
        # 检查buffer是否有足够的数据
        if len(self.buffers[self.agent_x]) < batch_size:
            return
        
        # 采样一次batch，所有agent共享（避免重复采样导致的不一致性）
        obs, action, reward, next_obs, done, next_action = self.sample(batch_size)
        
        if obs is None:
            return
        
        # 多智能体特有-- 集中式训练critic:计算next_q值时,要用到所有智能体next状态和动作
        # 对每个agent分别更新critic和actor
        for agent_id, agent in self.agents.items():
            # 使用target critic计算next Q值
            with torch.no_grad():
                next_target_Q = agent.critic_target(next_obs.values(), next_action.values())
            
            # 先更新critic
            target_Q = reward[agent_id] + gamma * next_target_Q * (1 - done[agent_id])
            current_Q = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            agent.update_critic(critic_loss)
            
            # 再更新actor
            new_action = agent.actor(obs[agent_id])
            # 构建包含新action的动作字典（注意：不要直接修改action字典，要创建新字典）
            action_for_actor = {}
            for k, v in action.items():
                if k == agent_id:
                    action_for_actor[k] = new_action
                else:
                    action_for_actor[k] = v
            actor_loss = -agent.critic(obs.values(), action_for_actor.values()).mean()
            
            if self.regular:  # 和DDPG.py中的weight_decay一样原理
                actor_loss += (new_action**2).mean() * 1e-3
            
            agent.update_actor(actor_loss)
        
        # 软更新target网络
        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

    def save(self, model_dir, name):
        """
        保存模型
        """
        save_data = {agent_id: agent.actor.state_dict() for agent_id, agent in self.agents.items()}
        # 可选：保存每个agent的critic
        for agent_id, agent in self.agents.items():
            save_data[f'{agent_id}_critic'] = agent.critic.state_dict()
        
        torch.save(
            save_data,
            os.path.join(model_dir, f'MADDPG_{name}_pay.pth')
        )

    @staticmethod
    def load(num_agents, obs_dim, action_dim, is_continue, model_dir, trick=None, horizon=1000, device='cpu'):
        """
        加载模型
        """
        # 使用指定的设备来初始化模型
        policy = MADDPG(num_agents, obs_dim, action_dim, is_continue=is_continue,
                       actor_lr=0, critic_lr=0, horizon=horizon, device=device, trick=trick)
        
        # 如果model_dir是文件路径，直接使用；否则拼接文件名
        if os.path.isfile(model_dir):
            model_path = model_dir
        else:
            # 尝试找到 MADDPG 模型文件
            model_files = [f for f in os.listdir(model_dir) if f.startswith('MADDPG_') and f.endswith('.pth')]
            if model_files:
                model_path = os.path.join(model_dir, model_files[0])
            else:
                raise FileNotFoundError(f"No MADDPG model found in {model_dir}")
        
        data = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        
        # 加载每个agent的actor模型
        for agent_id, agent in policy.agents.items():
            if agent_id in data:
                agent.actor.load_state_dict(data[agent_id])
                agent.actor_target.load_state_dict(data[agent_id])
            # 加载critic（如果存在）
            critic_key = f'{agent_id}_critic'
            if critic_key in data:
                agent.critic.load_state_dict(data[critic_key])
                agent.critic_target.load_state_dict(data[critic_key])
        
        return policy
