import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from algorithm.buffer import Buffer_for_PPO
from algorithm.MAPPO import Actor, Actor_discrete


class MADAPOAgent:
    """
    Per-agent policy for MADAPO. Reuses the actor architectures defined in MAPPO.
    """

    def __init__(self, obs_dim, action_dim, actor_lr, is_continue, device, trick):
        self.is_continue = is_continue
        self.device = device
        self.trick = trick or {}

        if is_continue:
            self.actor = Actor(obs_dim, action_dim, trick=self.trick).to(self.device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=self.trick).to(self.device)

        adam_eps = 1e-5 if self.trick.get("adam_eps", False) else 1e-8
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=adam_eps)

    def update(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def lr_decay(self, lr):
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = lr


class MADAPO:
    """
    Multi-agent DAPO/GRPO variant adapted to the microgrid environment.
    This class keeps MAPPO-style interfaces (select_action/add/learn) but
    adopts policy-only GRPO-style updates per agent.
    """

    def __init__(
        self,
        num_agents,
        obs_dim,
        action_dim,
        is_continue,
        actor_lr,
        horizon,
        device,
        trick=None,
    ):
        self.device = device
        self.is_continue = is_continue
        self.horizon = int(horizon)
        self.trick = trick or {}
        self.num_agents = num_agents

        self.agents = {}
        self.buffers = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = MADAPOAgent(obs_dim, action_dim, actor_lr, is_continue, device, self.trick)
            act_dim = action_dim if is_continue else 1
            self.buffers[agent_id] = Buffer_for_PPO(horizon, obs_dim, act_dim=act_dim, device=device)

        if self.trick.get("lr_decay", False):
            self.actor_lr = actor_lr

    def select_action(self, obs):
        actions = {}
        log_pis = {}
        for agent_id, agent_obs in obs.items():
            agent_obs_tensor = torch.as_tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.is_continue:
                mean, std = self.agents[agent_id].actor(agent_obs_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_pi = dist.log_prob(action)
            else:
                dist = Categorical(probs=self.agents[agent_id].actor(agent_obs_tensor))
                action = dist.sample()
                log_pi = dist.log_prob(action)

            actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
            log_pis[agent_id] = log_pi.detach().cpu().numpy().squeeze(0)

        return actions, log_pis

    def evaluate_action(self, obs):
        actions = {}
        for agent_id, agent_obs in obs.items():
            agent_obs_tensor = torch.as_tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.is_continue:
                mean, _ = self.agents[agent_id].actor(agent_obs_tensor)
                action = mean.detach().cpu().numpy().squeeze(0)
            else:
                probs = self.agents[agent_id].actor(agent_obs_tensor).detach()
                action = torch.argmax(probs, dim=-1).cpu().numpy().squeeze(0)
            actions[agent_id] = action
        return actions

    def add(self, obs, action, reward, next_obs, done, log_pi, adv_dones):
        for agent_id, buffer in self.buffers.items():
            buffer.add(
                obs[agent_id],
                action[agent_id],
                reward[agent_id],
                next_obs[agent_id],
                done[agent_id],
                log_pi[agent_id],
                adv_dones[agent_id],
            )

    def _compute_advantages(self, rewards, adv_dones, gamma):
        """
        GRPO-style episode return normalization (group normalization).
        """
        with torch.no_grad():
            rew_np = rewards.reshape(-1).cpu().numpy()
            adv_done_np = adv_dones.reshape(-1).cpu().numpy().astype(bool)

            episode_returns = []
            episode_ids = []
            ep_start = 0

            for t in range(len(rew_np)):
                terminal = adv_done_np[t] or (t == len(rew_np) - 1)
                if terminal:
                    G = 0.0
                    for k in range(t, ep_start - 1, -1):
                        G = rew_np[k] + gamma * G
                    episode_returns.append(G)
                    ep_idx = len(episode_returns) - 1
                    episode_ids.extend([ep_idx] * (t - ep_start + 1))
                    ep_start = t + 1

            episode_returns = torch.tensor(episode_returns, dtype=torch.float32, device=self.device)
            episode_ids = torch.tensor(episode_ids, dtype=torch.long, device=self.device)

            mean_R = episode_returns.mean()
            std_R = episode_returns.std(unbiased=False) + 1e-8
            A_ep = (episode_returns - mean_R) / std_R

            advantages = A_ep[episode_ids].unsqueeze(1)
            if self.trick.get("adv_norm", False):
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            adv_abs = advantages.abs().squeeze(1)
            valid_indices = torch.nonzero(adv_abs > 1e-6, as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                valid_indices = torch.arange(advantages.shape[0], device=self.device)

        return advantages, valid_indices

    def learn(self, minibatch_size, gamma, eps_clip_low, eps_clip_high, K_epochs, entropy_coef):
        for agent_id, buffer in self.buffers.items():
            obs, act, rew, _, _, logp_old, adv_dones = buffer.all()
            advantages, valid_indices = self._compute_advantages(rew, adv_dones, gamma)

            valid_indices_np = valid_indices.cpu().numpy()

            for _ in range(K_epochs):
                perm = np.random.permutation(valid_indices_np.shape[0])
                for start in range(0, valid_indices_np.shape[0], minibatch_size):
                    mb_indices = perm[start : start + minibatch_size]
                    if mb_indices.size == 0:
                        continue
                    mb_global = valid_indices_np[mb_indices]

                    obs_mb = obs[mb_global]
                    act_mb = act[mb_global]
                    old_logp_mb = logp_old[mb_global]
                    adv_mb = advantages[mb_global]

                    if old_logp_mb.ndim > 1:
                        old_logp_mb = old_logp_mb.sum(dim=1, keepdim=True)
                    else:
                        old_logp_mb = old_logp_mb.view(-1, 1)

                    agent = self.agents[agent_id]

                    if self.is_continue:
                        mean, std = agent.actor(obs_mb)
                        dist = Normal(mean, std)
                        logp_now = dist.log_prob(act_mb).sum(dim=1, keepdim=True)
                        entropy = dist.entropy().sum(dim=1, keepdim=True)
                    else:
                        act_mb_long = act_mb.view(-1).long()
                        dist = Categorical(probs=agent.actor(obs_mb))
                        logp_now = dist.log_prob(act_mb_long).unsqueeze(1)
                        entropy = dist.entropy().unsqueeze(1)

                    ratios = torch.exp(logp_now - old_logp_mb)
                    surr1 = ratios * adv_mb
                    clipped = torch.clamp(ratios, 1.0 - eps_clip_low, 1.0 + eps_clip_high)
                    surr2 = clipped * adv_mb

                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy_coef * entropy.mean()
                    loss = policy_loss + entropy_loss

                    agent.update(loss)

            buffer.clear()

    def lr_decay(self, episode_num, max_episodes):
        if not self.trick.get("lr_decay", False):
            return

        lr_now = self.actor_lr * (1 - episode_num / max_episodes)
        for agent in self.agents.values():
            agent.lr_decay(lr_now)


