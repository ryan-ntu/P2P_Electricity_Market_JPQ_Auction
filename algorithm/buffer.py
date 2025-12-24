import torch
import numpy as np


class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children
    Used to efficiently sample based on priority.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Number of tree nodes: 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Actual stored transitions
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        """Add a new sample with priority based on error"""
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        """Sample a batch of n transitions with importance-sampling weights"""
        batch = []
        idxs = []
        segment = self.tree.total / n
        priorities = []

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        for i in range(n):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        return idxs, batch, is_weights

    def update(self, idxs, errors):
        """Update priorities of sampled transitions after learning from them"""
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

class Buffer_for_PPO:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device, trick = None):
        self.capacity = capacity = int(capacity)

        self.obs = np.zeros((capacity, obs_dim))    # batch_size x state_dim
        self.actions = np.zeros((capacity, act_dim))  # batch_size x action_dim
        self.rewards = np.zeros(capacity)            # just a tensor with length: batch
        self.next_obs = np.zeros((capacity, obs_dim))  # batch_size x state_dim
        self.dones = np.zeros(capacity, dtype=bool)    # just a tensor with length: batch_size
        if trick is not None and trick['decaystd']:
            self.action_log_probs = np.zeros((capacity)) 
        else:
            self.action_log_probs = np.zeros((capacity, act_dim)) 
        self.adv_dones = np.zeros(capacity, dtype=bool)    

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done, action_log_probs, adv_done):
        """ add an experience to the memory """ #
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        self.action_log_probs[self._index] = action_log_probs
        self.adv_dones[self._index] = adv_done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    # __len__ is a magic method in Python 可以让对象实现len()方法
    def __len__(self):
        return self._size
    
    # 清空buffer
    def clear(self):
        self._index = 0
        self._size = 0

    def all(self):
        obs = torch.as_tensor(self.obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        actions = torch.as_tensor(self.actions,dtype=torch.float32).to(self.device) # torch.Size([batch_size, action_dim])
        rewards = torch.as_tensor(self.rewards,dtype=torch.float32).reshape(-1,1).to(self.device)  # torch.Size([batch_size]) -> torch.Size([batch_size, 1])
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.as_tensor(self.next_obs,dtype=torch.float32).to(self.device)  # torch.Size([batch_size, state_dim])
        
        dones = torch.as_tensor(self.dones,dtype=torch.float32).reshape(-1,1).to(self.device)

        actions_log_probs = torch.as_tensor(self.action_log_probs,dtype=torch.float32).to(self.device) # torch.Size([batch_size, action_dim])
        adv_dones = torch.as_tensor(self.adv_dones,dtype=torch.float32).reshape(-1,1).to(self.device)
        
        return obs, actions, rewards, next_obs, dones , actions_log_probs, adv_dones
