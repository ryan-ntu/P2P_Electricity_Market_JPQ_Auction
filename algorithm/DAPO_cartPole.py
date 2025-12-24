import os
# 让空闲线程不占用 CPU（可选）
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from Buffer import Buffer_for_PPO


# =====================
# 一些小工具
# =====================

def orthogonal_init(layer, gain=1.0):
    """
    正交初始化：常用于 PPO / A2C 等，提高训练稳定性
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


# =====================
# Actor 网络（离散动作）
# =====================

class ActorDiscrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128, trick=None):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

        self.trick = trick or {}
        if self.trick.get('orthogonal_init', False):
            orthogonal_init(self.l1)
            orthogonal_init(self.l2)
            orthogonal_init(self.l3, gain=0.01)

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        logits = self.l3(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def get_dist(self, obs):
        probs = self.forward(obs)
        return Categorical(probs=probs)


# =====================
# 纯 policy-only 的 DAPO 算法
# =====================

class DAPO:
    """
    纯 policy-only 的 DAPO：
    - 不使用 Critic / Value 网络
    - advantage = GRPO-style 的 episode return group-normalization
    - 使用不对称 clip: [1 - eps_clip_low, 1 + eps_clip_high]
    - 简化版 dynamic sampling：只在 |advantage| 较大的时间步上更新
    """
    def __init__(self, dim_info, actor_lr, horizon, device, trick=None):
        obs_dim, action_dim = dim_info
        self.device = device
        self.horizon = int(horizon)

        self.trick = trick or {}
        self.actor = ActorDiscrete(obs_dim, action_dim, trick=self.trick).to(self.device)
        # 使用 Buffer_for_PPO，对于离散动作空间 act_dim=1
        self.buffer = Buffer_for_PPO(horizon, obs_dim, act_dim=1, device=device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            eps=1e-5 if self.trick.get('adam_eps', False) else 1e-8,
        )

        if self.trick.get('lr_decay', False):
            self.actor_lr = actor_lr

    # -------- 交互相关 --------
    def select_action(self, obs):
        """
        obs: np.array (obs_dim,)
        return: action (int), logp (float)
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.get_dist(obs_tensor)
        action = dist.sample()            # (1,)
        logp = dist.log_prob(action)      # (1,)

        action_np = int(action.item())
        logp_np = float(logp.item())
        return action_np, logp_np

    def evaluate_action(self, obs):
        """
        用于测试：选择概率最大的动作（贪心）
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.get_dist(obs_tensor)
        action = torch.argmax(dist.probs, dim=-1)
        return int(action.item())

    def add(self, obs, act, rew, next_obs, done, logp, adv_done):
        """
        添加经验到 buffer
        obs: np.array (obs_dim,)
        act: int
        rew: float
        next_obs: np.array (obs_dim,)
        done: bool (terminated)
        logp: float (log π(a|s) at sampling time)
        adv_done: bool (terminated or truncated，用于 advantage 计算)
        """
        # 确保 action 和 action_log_pi 是数组形式（对于离散动作空间，act_dim=1）
        if isinstance(act, (int, np.integer)):
            act = np.array([act])
        elif isinstance(act, np.ndarray) and act.ndim == 0:
            act = np.array([act.item()])
        elif isinstance(act, np.ndarray) and act.ndim > 0:
            act = act.flatten()
        
        if isinstance(logp, (int, float)):
            logp = np.array([logp])
        elif isinstance(logp, np.ndarray) and logp.ndim == 0:
            logp = np.array([logp.item()])
        elif isinstance(logp, np.ndarray) and logp.ndim > 0:
            logp = logp.flatten()
        
        self.buffer.add(obs, act, rew, next_obs, done, logp, adv_done)

    # -------- DAPO 学习部分（无 critic） --------
    def learn(self, minibatch_size, gamma, eps_clip_low, eps_clip_high, K_epochs, entropy_coef):
        """
        关键点：
        1. 按 episode 划分，计算每条 episode 的 discounted return
        2. 用所有 episode return 做 group-normalization 得到 A_ep
        3. 映射到每个时间步：adv[t] = A_ep[episode_id[t]]
        4. 使用 PPO-style ratio + DAPO-style asymmetric clip 更新 policy
        5. 简化 dynamic sampling：只在 |adv| 较大的时间步上进行更新
        """
        obs, act, rew, next_obs, done, logp_old, adv_dones = self.buffer.all()
        # 对于离散动作空间，需要将 actions 转换为 long 类型
        act = act.long()  # (T, act_dim)
        T = obs.shape[0]  # 不直接用 self.horizon，防止未来 buffer 长度不完全一致

        # ---- 计算每条 episode 的 discounted return（GRPO-style）----
        with torch.no_grad():
            rew_np = rew.reshape(-1).cpu().numpy()
            adv_dones_np = adv_dones.reshape(-1).cpu().numpy().astype(bool)

            episode_returns = []
            episode_ids = []
            ep_start = 0

            for t in range(T):
                is_terminal = adv_dones_np[t] or (t == T - 1)
                if is_terminal:
                    # 折扣回报（从 t 回溯到 ep_start）
                    G = 0.0
                    for k in range(t, ep_start - 1, -1):
                        G = rew_np[k] + gamma * G
                    episode_returns.append(G)

                    # 这段时间步属于同一条 episode
                    ep_idx = len(episode_returns) - 1
                    episode_ids.extend([ep_idx] * (t - ep_start + 1))

                    ep_start = t + 1

            episode_returns = torch.tensor(episode_returns, dtype=torch.float32, device=self.device)  # (E,)
            episode_ids = torch.tensor(episode_ids, dtype=torch.long, device=self.device)            # (T,)

            # ---- GRPO-style group normalization ----
            mean_R = episode_returns.mean()
            std_R = episode_returns.std() + 1e-8
            A_ep = (episode_returns - mean_R) / std_R   # (E,)

            # 映射到每个 time-step：advantages[t] = A_ep[episode_ids[t]]
            advantages = A_ep[episode_ids]  # (T,)
            advantages = advantages.unsqueeze(1)  # (T,1)

            # 可选：adv 再标准化
            if self.trick.get('adv_norm', False):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---- 简化版 dynamic sampling：只在 |adv| 足够大的时间步上更新 ----
            adv_abs = advantages.abs().squeeze(1)  # (T,)
            adv_eps = 1e-6
            valid_mask = (adv_abs > adv_eps)
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)  # shape (N_valid,)

            # 如果所有 advantage 几乎都为 0，则退化为用全部样本，避免没有梯度
            if valid_indices.numel() == 0:
                valid_indices = torch.arange(T, device=self.device)

        # ---- DAPO: 多 epoch, mini-batch 更新 actor（无 critic）----
        valid_indices_np = valid_indices.cpu().numpy()

        for _ in range(K_epochs):
            # 只在 valid_indices 上做 shuffle 和 mini-batch
            perm = np.random.permutation(valid_indices_np.shape[0])
            for start in range(0, valid_indices_np.shape[0], minibatch_size):
                end = start + minibatch_size
                mb_id = perm[start:end]
                mb_idx = valid_indices_np[mb_id]   # 这些是全局时间步索引

                obs_mb = obs[mb_idx]                  # (B, obs_dim)
                act_mb = act[mb_idx]                  # (B, act_dim) 对于离散动作空间 act_dim=1
                old_logp_mb = logp_old[mb_idx]
                if old_logp_mb.ndim > 1:
                    old_logp_mb = old_logp_mb.sum(dim=1, keepdim=True)  # 对连续动作空间求和
                else:
                    old_logp_mb = old_logp_mb.reshape(-1, 1)  # (B, 1)
                adv_mb = advantages[mb_idx]           # (B,1)

                dist_now = self.actor.get_dist(obs_mb)
                # act_mb 从 buffer 中取出时形状是 (B, act_dim)，对于离散动作空间 act_dim=1
                # 需要 reshape 为 (B,) 用于 log_prob
                act_mb_flat = act_mb.reshape(-1) if act_mb.ndim > 1 else act_mb
                logp_now = dist_now.log_prob(act_mb_flat).unsqueeze(1)  # (B,1)
                entropy_mb = dist_now.entropy().unsqueeze(1)       # (B,1)

                ratios = torch.exp(logp_now - old_logp_mb)         # (B,1)

                # DAPO: 不对称 clip
                surr1 = ratios * adv_mb
                clipped_ratios = torch.clamp(ratios,
                                             1.0 - eps_clip_low,
                                             1.0 + eps_clip_high)
                surr2 = clipped_ratios * adv_mb

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_coef * entropy_mb.mean()
                loss = policy_loss + entropy_loss

                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

        self.buffer.clear()

    def lr_decay(self, episode_num, max_episodes):
        lr_now = self.actor_lr * (1 - episode_num / max_episodes)
        for g in self.actor_optimizer.param_groups:
            g['lr'] = lr_now


# =====================
# Env 工具函数
# =====================

def get_env(env_name):
    env = gym.make(env_name)
    if isinstance(env.observation_space, gym.spaces.Box):
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = 1

    if isinstance(env.action_space, gym.spaces.Box):
        raise NotImplementedError("当前示例只实现了离散动作空间。")
    else:
        action_dim = env.action_space.n
        dim_info = [obs_dim, action_dim]
    return env, dim_info


# =====================
# 主函数：训练 CartPole-v1
# =====================

if __name__ == "__main__":
    env_name = "CartPole-v1"
    seed = 0
    max_episodes = 500

    gamma = 0.99
    actor_lr = 1e-3
    horizon = 2048

    # DAPO 不对称 clip 参数（你可以尝试改成 0.2 / 0.3 等）
    eps_clip_low = 0.3
    eps_clip_high = 0.3

    K_epochs = 10 
    entropy_coef = 0.01
    minibatch_size = 128

    trick = {
        "adv_norm": True,
        "orthogonal_init": True,
        "adam_eps": False,
        "lr_decay": False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env, dim_info = get_env(env_name)
    obs_dim, action_dim = dim_info

    print(f"Env: {env_name}, obs_dim={obs_dim}, action_dim={action_dim}, device={device}")
    print(f"DAPO(policy-only) params: eps_low={eps_clip_low}, eps_high={eps_clip_high}")

    algo = DAPO(dim_info, actor_lr, horizon, device, trick=trick)

    episode_num = 0
    step_count = 0
    episode_return = 0.0
    returns = []

    obs, info = env.reset(seed=seed)

    while episode_num < max_episodes:
        step_count += 1

        action, logp = algo.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        adv_done = terminated or truncated  # adv_done 用于 advantage 计算

        algo.add(obs, action, reward, next_obs, done, logp, adv_done)

        episode_return += reward
        obs = next_obs

        if done:
            episode_num += 1
            returns.append(episode_return)
            if episode_num % 100 == 0:
                avg_ret = np.mean(returns[-10:])
                print(f"Episode {episode_num}, Return={episode_return:.2f}, Avg(10)={avg_ret:.2f}")
            obs, info = env.reset()
            episode_return = 0.0

        if len(algo.buffer) == algo.horizon:
            algo.learn(minibatch_size, gamma, eps_clip_low, eps_clip_high, K_epochs, entropy_coef)
            if trick.get("lr_decay", False):
                algo.lr_decay(episode_num, max_episodes)

        # 简单 early stopping
