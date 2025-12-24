import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from copy import deepcopy
from algorithm.buffer import Buffer_for_DDPG
import numpy as np
import os


class SimpleGCNLayer(nn.Module):
    def __init__(self, num_agents: int, input_dim: int, hidden_dim: int):
        super(SimpleGCNLayer, self).__init__()
        self.num_agents = num_agents
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

        adj = torch.ones(num_agents, num_agents, dtype=torch.float32)
        self.register_buffer("adjacency", self._normalize_adj(adj))

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(adj.size(0), device=adj.device)
        a_hat = adj + eye
        degree = torch.sum(a_hat, dim=1)
        degree = torch.where(degree == 0, torch.ones_like(degree), degree)
        d_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        return d_inv_sqrt @ a_hat @ d_inv_sqrt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, input_dim)
        agg = torch.einsum("ij,bjd->bid", self.adjacency, x)
        h = F.relu(self.fc1(agg))
        h = self.fc2(h)
        return h + x  # residual


class TemporalLinear(nn.Module):
    def __init__(self, seq_len: int):
        super(TemporalLinear, self).__init__()
        self.linear = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F, L)
        b, n, f, l = x.shape
        x = x.reshape(b * n * f, l)
        x = self.linear(x)
        x = x.reshape(b, n, f, l)
        return x


class GraphTemporalEncoder(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int, temporal_len: int = 28, seq_len: int = 7, gcn_hidden: int = 256):
        super(GraphTemporalEncoder, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.temporal_len = min(max(temporal_len, 0), obs_dim)
        self.seq_len = seq_len
        self.temporal_start = 3 if obs_dim - self.temporal_len >= 5 else 0
        self.temporal_len = min(self.temporal_len, obs_dim - self.temporal_start)
        self.feature_dim = self.temporal_len // self.seq_len if self.seq_len > 0 else 0

        self.gcn = SimpleGCNLayer(num_agents, obs_dim, gcn_hidden)
        self.temporal_linear = TemporalLinear(self.seq_len) if self.temporal_len > 0 and self.seq_len > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, obs_dim)
        gcn_out = self.gcn(x)
        if self.temporal_linear is None or self.temporal_len == 0:
            return gcn_out

        start = self.temporal_start
        end = start + self.temporal_len
        temporal = gcn_out[:, :, start:end]
        if self.feature_dim == 0:
            return gcn_out

        temporal = temporal.reshape(temporal.shape[0], temporal.shape[1], self.feature_dim, self.seq_len)
        temporal = self.temporal_linear(temporal)
        temporal = temporal.reshape(temporal.shape[0], temporal.shape[1], -1)

        encoded = torch.cat(
            [
                gcn_out[:, :, :start],
                temporal,
                gcn_out[:, :, end:],
            ],
            dim=2,
        )
        return encoded


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
        global_obs_act_dim = obs_dim * num_agents + action_dim * num_agents

        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, s, a):
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
        default_trick = {
            "orthogonal_init": False,
            "adam_eps": False,
            "LayerNorm": False,
            "feature_norm": False,
        }
        if trick is None:
            trick = default_trick.copy()
        else:
            for k, v in default_trick.items():
                trick.setdefault(k, v)

        self.agents = {}
        self.buffers = {}
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]

        self.encoder = GraphTemporalEncoder(num_agents, obs_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=actor_lr)

        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = Agent(obs_dim, action_dim, num_agents, actor_lr, critic_lr, device, trick)
            self.buffers[agent_id] = Buffer_for_DDPG(horizon, obs_dim, action_dim, device)

        self.device = device
        self.is_continue = is_continue
        self.agent_x = self.agent_ids[0]
        self.regular = False
        self.trick = trick

        print("MADDPG actor_type: continue") if self.is_continue else print("MADDPG actor_type: discrete")

    def _ensure_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def _encode_obs_single(self, obs_dict):
        tensors = []
        for agent_id in self.agent_ids:
            tensor = self._ensure_tensor(obs_dict[agent_id])
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        stacked = torch.stack(tensors, dim=1)  # (1, N, obs_dim)
        encoded = self.encoder(stacked)
        return {
            agent_id: encoded[:, idx, :]
            for idx, agent_id in enumerate(self.agent_ids)
        }

    def _encode_obs_batch(self, obs_dict):
        tensors = []
        batch = None
        for agent_id in self.agent_ids:
            tensor = self._ensure_tensor(obs_dict[agent_id])
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if batch is None:
                batch = tensor.shape[0]
            tensors.append(tensor)
        stacked = torch.stack(tensors, dim=1)  # (B, N, obs_dim)
        encoded = self.encoder(stacked)
        return {
            agent_id: encoded[:, idx, :]
            for idx, agent_id in enumerate(self.agent_ids)
        }

    def select_action(self, obs):
        actions = {}
        log_probs = {}
        encoded_obs = self._encode_obs_single(obs)
        for agent_id in self.agent_ids:
            obs_tensor = encoded_obs[agent_id]
            if self.is_continue:
                action = self.agents[agent_id].actor(obs_tensor)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
                log_probs[agent_id] = None
            else:
                raise NotImplementedError("Discrete action space not implemented")
        return actions, log_probs

    def evaluate_action(self, obs):
        actions = {}
        encoded_obs = self._encode_obs_single(obs)
        for agent_id in self.agent_ids:
            obs_tensor = encoded_obs[agent_id]
            if self.is_continue:
                action = self.agents[agent_id].actor(obs_tensor)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
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
            return None, None, None, None, None
        if total_size < batch_size:
            batch_size = total_size
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)

        return obs, action, reward, next_obs, done

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
        
        obs, action, reward, next_obs, done = self.sample(batch_size)
        
        if obs is None:
            return
        
        self.encoder_optimizer.zero_grad(set_to_none=True)

        for agent_id, agent in self.agents.items():
            encoded_obs = self._encode_obs_batch(obs)
            encoded_next_obs = self._encode_obs_batch(next_obs)
            with torch.no_grad():
                next_actions = {
                    aid: self.agents[aid].actor_target(encoded_next_obs[aid])
                    for aid in self.agent_ids
                }
                next_target_Q = agent.critic_target(encoded_next_obs.values(), next_actions.values())

            target_Q = reward[agent_id] + gamma * next_target_Q * (1 - done[agent_id])
            current_Q = agent.critic(encoded_obs.values(), action.values())
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            agent.update_critic(critic_loss)

            encoded_obs_actor = self._encode_obs_batch(obs)
            new_action = agent.actor(encoded_obs_actor[agent_id])
            action_for_actor = {}
            for k, v in action.items():
                action_for_actor[k] = new_action if k == agent_id else v
            actor_loss = -agent.critic(encoded_obs_actor.values(), action_for_actor.values()).mean()

            if self.regular:
                actor_loss += (new_action ** 2).mean() * 1e-3

            agent.update_actor(actor_loss)

        self.encoder_optimizer.step()
        
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
