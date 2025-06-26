"""
Deep Deterministic Policy Gradient (DDPG) Agent Implementation
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from .networks import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    """
    DDPG Agent with Actor network and now supports Twin Critics for TD3.
    """
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), 
                 actor_lr=1e-4, critic_lr=1e-3, tau=1e-3,
                 centralized=False, total_state_size=None, total_action_size=None,
                 action_low=None, action_high=None):
        """
        Initialize a DDPG agent.
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
            hidden_sizes (tuple): Sizes of hidden layers for networks
            actor_lr (float): Learning rate for the actor
            critic_lr (float): Learning rate for the critic
            tau (float): Soft update parameter
            centralized (bool): Whether to use centralized critic
            total_state_size (int): Total dimension of all agents' states (for centralized critic)
            total_action_size (int): Total dimension of all agents' actions (for centralized critic)
            action_low (float or array): Lower bound of the action space (default: -1.0)
            action_high (float or array): Upper bound of the action space (default: 1.0)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.centralized = centralized
        
        # Set action bounds
        self.action_low = -1.0 if action_low is None else action_low
        self.action_high = 1.0 if action_high is None else action_high
        self.action_range = self.action_high - self.action_low
        
        # Actor Networks (Local and Target)
        self.actor = Actor(state_size, action_size, hidden_sizes, 
                          action_low=self.action_low, 
                          action_high=self.action_high).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_sizes,
                                 action_low=self.action_low, 
                                 action_high=self.action_high).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Define critic input sizes based on centralized or decentralized setting
        if centralized:
            critic_state_size = total_state_size
            critic_action_size = total_action_size
        else:
            critic_state_size = state_size
            critic_action_size = action_size
            
        # Critic 1 Networks (Local and Target)
        self.critic = Critic(critic_state_size, critic_action_size, hidden_sizes).to(device)
        self.critic_target = Critic(critic_state_size, critic_action_size, hidden_sizes).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Critic 2 Networks (Local and Target) - For TD3
        self.critic2 = Critic(critic_state_size, critic_action_size, hidden_sizes).to(device)
        self.critic_target2 = Critic(critic_state_size, critic_action_size, hidden_sizes).to(device)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Initialize target networks with local network weights
        self.hard_update(self.critic_target, self.critic)
        self.hard_update(self.critic_target2, self.critic2)
        self.hard_update(self.actor_target, self.actor)
        
    def act(self, state, add_noise=True, noise_scale=1.0):
        """
        Returns actions for given state as per current policy.
        
        Args:
            state: Current state
            add_noise (bool): Whether to add noise for exploration
            noise_scale (float): Scale factor for noise
        """
        state = torch.from_numpy(state).float().to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        
        if add_noise:
            scaled_noise = np.random.normal(0, noise_scale * self.action_range, size=action.shape)
            action += scaled_noise
            
        return np.clip(action, self.action_low, self.action_high)
    
    def act_target(self, state):
        """
        Returns actions for given state as per current target policy.
        Keeps gradients for learning.
        
        Args:
            state: Current state (tensor)
            
        Returns:
            action: Action from target policy (tensor)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        return self.actor_target(state)
    
    def hard_update(self, target_model, source_model):
        """
        Hard update model parameters.
        θ_target = θ_source
        """
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(source_param.data)
        
    def soft_update(self, target, source):
        """
        Soft update model parameters.
        θ_target = τ*θ_source + (1 - τ)*θ_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    # Note: The 'learn' method is removed from here as the learning logic
    # is handled by the main MADDPG/MATD3 class for centralized training.
    # The 'save' and 'load' methods are also better handled by the main class.