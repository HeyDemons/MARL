import numpy as np
import torch

# 经验回放池
class ReplayBuffer:

    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._size = 0
        self._index= 0
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        self._size = min(self._size + 1, self.capacity)
        self._index = (self._index + 1) % self.capacity
    def sample(self, indices):
        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_obs[indices]
        dones = self.dones[indices]
        
        obs = torch.from_numpy(obs).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        return obs, actions, rewards, next_obs, dones
    def __len__(self):
        return self._size
        