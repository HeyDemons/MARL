import numpy as np
import torch

class ReplayBuffer:
    """
    A simple replay buffer for storing transitions in a multi-agent setting.
    """
    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=bool)

        self._index = 0
        self._size = 0
        self.device = device
    def add(self, obs, action, reward, next_obs, done):
        """
        Add a transition to the replay buffer.
        """
        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.dones[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        obs = self.obs[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_obs[indices]
        dones = self.dones[indices]
        return (torch.tensor(obs, dtype=torch.float32, device=self.device),
                torch.tensor(actions, dtype=torch.float32, device=self.device),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(next_obs, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.bool, device=self.device))
    
    def __len__(self):
        return self._size
