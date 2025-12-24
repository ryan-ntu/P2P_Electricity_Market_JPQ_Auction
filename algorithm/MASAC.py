import os

os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from copy import deepcopy

from algorithm.buffer import Buffer
from algorithm.buffer import Buffer_for_DDPG
'''
这里实现了8种MASAC的写法，
与论文mSAC不同：mSAC 还加入了分解值算法和反事实曲线 mSAC论文链接：https://arxiv.org/pdf/2104.06655

实验发现：效果 
random_steps = 0时 > random_steps = 500
log_std 法1 > log_std 法2 见note中结果
这里选用上述两者较好的结果
'''




## 第一部分：定义Agent类
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim) 
        self.log_std_layer = nn.Linear(hidden_2, action_dim) # 此方法可改为MAPPO中只训练一个std的方法  ## log_std 法1
        ##self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # 与PPO.py的方法一致：对角高斯函数  ## log_std 法2

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # 我们输出log_std以确保std=exp(log_std)>0 ## log_std 法1
        ##log_std = self.log_std.expand_as(mean)  ## log_std 法2
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # 生成一个高斯分布
        if deterministic:  # 评估时用
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # 方法参考Open AI Spinning up，更稳定。见https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L53C12-L53C24
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True) # batch_size x 1
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True) #这里是计算tanh的对数概率，
        else: #常见的其他写法
            '''
            log_pi =  dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= torch.log(1 - torch.tanh(a).pow(2) + 1e-6).sum(dim=1, keepdim=True) # 1e-6是为了数值稳定性 
            '''
            log_pi = None
        
        a =  torch.tanh(a)  # 使用tanh将无界的高斯分布压缩到有界的动作区间内。

        return a, log_pi
'''
集中式训练Critic
'''    
class Critic(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, hidden_1=128, hidden_2=128):
        super(Critic, self).__init__()
        global_obs_act_dim = obs_dim * num_agents + action_dim * num_agents
        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        self.l1_2 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2_2 = nn.Linear(hidden_1, hidden_2)
        self.l3_2 = nn.Linear(hidden_2, 1)


    def forward(self, s, a):
        sa = torch.cat(list(s) + list(a), dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l1_2(sa))
        q2 = F.relu(self.l2_2(q2))
        q2 = self.l3_2(q2)
        return q1, q2
    
class Agent:
    def __init__(self, obs_dim, action_dim, num_agents, actor_lr, critic_lr, device, trick=None):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(num_agents, obs_dim, action_dim).to(device)

        if trick and trick.get("adam_eps", False):
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

class Alpha:
    def __init__(self, action_dim, alpha_lr= 0.0001, alpha = 0.01,requires_grad = False,is_continue = True):
        
        self.log_alpha = torch.tensor(np.log(alpha),dtype = torch.float32, requires_grad=requires_grad) # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        if is_continue:
            self.target_entropy = -action_dim # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper(SAC) 参考原sac论文
        else:
            self.target_entropy =  0.6 * (-torch.log(torch.tensor(1.0 / action_dim))) # 参考:https://zhuanlan.zhihu.com/p/566722896
        self.alpha = self.log_alpha.exp() # 更新actor时无detach会报错,是因为这里只有一个计算图 

    def update_alpha(self, loss):
        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

class MASAC:
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

        self.attention = False
        self.agents = {}
        self.buffers = {}
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        for agent_id in self.agent_ids:
            self.agents[agent_id] = Agent(obs_dim, action_dim, num_agents, actor_lr, critic_lr, device=device, trick=trick)
            self.buffers[agent_id] = Buffer_for_DDPG(horizon, obs_dim, action_dim, device)

        self.adaptive_alpha = True
        self.alphas = {}
        for agent_id in self.agent_ids:
            if self.adaptive_alpha:
                self.alphas[agent_id] = Alpha(action_dim, alpha=0.01, requires_grad=True, is_continue=is_continue)
            else:
                self.alphas[agent_id] = Alpha(action_dim, alpha=0.1, requires_grad=False, is_continue=is_continue)

        self.entropy_way_c = "1"
        self.entropy_way_a = "1"
        self.action_way = "1"
        self.device = device
        self.is_continue = is_continue
        self.agent_x = self.agent_ids[0]

        print("MASAC actor_type: continue") if self.is_continue else print("MASAC actor_type: discrete")
    
    def select_action(self, obs):
        actions = {}
        log_probs = {}
        for agent_id, obs_val in obs.items():
            obs_tensor = torch.as_tensor(obs_val, dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue:
                action, log_pi = self.agents[agent_id].actor(obs_tensor)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
                log_probs[agent_id] = log_pi.detach().cpu().numpy()
            else:
                raise NotImplementedError("Discrete action space not implemented")
        return actions, log_probs
    
    def evaluate_action(self, obs):
        actions = {}
        for agent_id, obs_val in obs.items():
            obs_tensor = torch.as_tensor(obs_val, dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue:
                action, _ = self.agents[agent_id].actor(obs_tensor, deterministic=True)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
            else:
                raise NotImplementedError("Discrete action space not implemented")
        return actions
    
    def add(self, obs, action, reward, next_obs, done, log_probs=None, adv_done=None):
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
        obs, action, reward, next_obs, done = self.sample(batch_size)
        if obs is None:
            return

        for agent_id, agent in self.agents.items():
            next_action = {}
            next_log_pi = {}
            with torch.no_grad():
                for aid in self.agent_ids:
                    n_act, n_log_pi = self.agents[aid].actor_target(next_obs[aid])
                    next_action[aid] = n_act
                    next_log_pi[aid] = n_log_pi

                q1_next_target, q2_next_target = agent.critic_target(next_obs.values(), next_action.values())
                q_next_target = torch.min(q1_next_target, q2_next_target)

                if self.entropy_way_c == "0":
                    stacked_log_pi = torch.stack(
                        [next_log_pi[aid] for aid in self.agent_ids], dim=1
                    ).sum(dim=1)
                    entropy_next = -stacked_log_pi
                else:
                    entropy_next = -next_log_pi[agent_id]

                q_target = reward[agent_id] + gamma * (1 - done[agent_id]) * (
                    q_next_target + self.alphas[agent_id].alpha.detach() * entropy_next
                )

            q1, q2 = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(q1, q_target.detach()) + F.mse_loss(q2, q_target.detach())
            agent.update_critic(critic_loss)

            log_pi_dict = {}
            if self.action_way == "0":
                new_action, log_pi = self.agents[agent_id].actor(obs[agent_id])
                action_current = dict(action)
                action_current[agent_id] = new_action
                q1_pi, q2_pi = agent.critic(obs.values(), action_current.values())
                log_pi_dict[agent_id] = log_pi
            else:
                action_current = {}
                for aid in self.agent_ids:
                    a, log_pi = self.agents[aid].actor(obs[aid])
                    action_current[aid] = a
                    log_pi_dict[aid] = log_pi
                q1_pi, q2_pi = agent.critic(obs.values(), action_current.values())

            if self.entropy_way_a == "0":
                stacked_log_pi = torch.stack([log_pi_dict.setdefault(aid, self.agents[aid].actor(obs[aid])[1]) for aid in self.agent_ids], dim=1).sum(dim=1)
                entropy = -stacked_log_pi
            else:
                if agent_id not in log_pi_dict:
                    _, log_pi_dict[agent_id] = self.agents[agent_id].actor(obs[agent_id])
                entropy = -log_pi_dict[agent_id]

            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (-q_pi - self.alphas[agent_id].alpha.detach() * entropy).mean()
            agent.update_actor(actor_loss)

            if self.adaptive_alpha:
                alpha_loss = (
                    self.alphas[agent_id].alpha
                    * (entropy - self.alphas[agent_id].target_entropy).detach()
                ).mean()
                self.alphas[agent_id].update_alpha(alpha_loss)

        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)

    def save(self, model_dir, name):
        save_data = {agent_id: agent.actor.state_dict() for agent_id, agent in self.agents.items()}
        for agent_id, agent in self.agents.items():
            save_data[f"{agent_id}_critic"] = agent.critic.state_dict()
        torch.save(save_data, os.path.join(model_dir, f"MASAC_{name}_pay.pth"))

    @staticmethod
    def load(num_agents, obs_dim, action_dim, is_continue, model_dir, trick=None, horizon=1000, device="cpu"):
        policy = MASAC(
            num_agents,
            obs_dim,
            action_dim,
            is_continue=is_continue,
            actor_lr=0,
            critic_lr=0,
            horizon=horizon,
            device=device,
            trick=trick,
        )

        if os.path.isfile(model_dir):
            model_path = model_dir
        else:
            model_files = [f for f in os.listdir(model_dir) if f.startswith("MASAC_") and f.endswith(".pth")]
            if model_files:
                model_path = os.path.join(model_dir, model_files[0])
            else:
                raise FileNotFoundError(f"No MASAC model found in {model_dir}")

        data = torch.load(model_path, map_location=torch.device(device), weights_only=False)

        for agent_id, agent in policy.agents.items():
            if agent_id in data:
                agent.actor.load_state_dict(data[agent_id])
                agent.actor_target.load_state_dict(data[agent_id])
            critic_key = f"{agent_id}_critic"
            if critic_key in data:
                agent.critic.load_state_dict(data[critic_key])
                agent.critic_target.load_state_dict(data[critic_key])

        return policy
