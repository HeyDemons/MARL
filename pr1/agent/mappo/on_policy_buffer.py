import torch
import numpy as np

class OnPolicyBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device, gamma, gae_lambda):
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate memory for the buffer
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self._index = 0
        self._size = 0

    def add(self, obs, action, reward, done, log_prob, value):
        if self._index >= self.capacity:
            self._index = 0

        self.obs[self._index] = obs
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.dones[self._index] = done
        self.log_probs[self._index] = log_prob
        self.values[self._index] = value
        
        self._index += 1
        self._size = min(self._index, self.capacity)

    def compute_advantages_and_returns(self, last_value, done):
        """
        Calculates the advantages and returns using Generalized Advantage Estimation (GAE).
        """
        advantage = 0
        for t in reversed(range(self._size)):
            if t == self._size - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t] # Use done from the next step in buffer
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            self.advantages[t] = advantage
        
        self.returns = self.advantages + self.values
        # Normalize advantages for stability
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


    def get_data(self):
        """
        Returns all stored data as torch tensors.
        """
        if self._size == 0:
            return None
            
        # Ensure we only return valid data
        indices = np.arange(self._size)

        obs = torch.from_numpy(self.obs[indices]).float().to(self.device)
        actions = torch.from_numpy(self.actions[indices]).float().to(self.device)
        log_probs = torch.from_numpy(self.log_probs[indices]).float().to(self.device)
        advantages = torch.from_numpy(self.advantages[indices]).float().to(self.device)
        returns = torch.from_numpy(self.returns[indices]).float().to(self.device)
        
        return obs, actions, log_probs, advantages, returns

    def clear(self):
        self._index = 0
        self._size = 0
    
    def __len__(self):
        return self._size