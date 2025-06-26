# 文件: matd3.py 
"""
Multi-Agent Twin-Delayed DDPG (MATD3) Implementation
"""
import torch
import torch.nn.functional as F
import numpy as np
from .ddpg import DDPGAgent
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MATD3:
    """
    Multi-Agent Twin-Delayed DDPG (MATD3) implementation
    """
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64), 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3, 
                 action_low=-1.0, action_high=1.0,
                 policy_update_freq=2, target_noise=0.2, noise_clip=0.5):
        """
        Initialize a MATD3 agent.
        
        Args:
            state_sizes (list): List of state sizes for each agent
            action_sizes (list): List of action sizes for each agent
            hidden_sizes (tuple): Sizes of hidden layers for networks
            actor_lr (float): Learning rate for the actor
            critic_lr (float): Learning rate for the critic
            gamma (float): Discount factor
            tau (float): Soft update parameter
            action_low (float or array): Lower bound of the action space
            action_high (float or array): Upper bound of the action space
            policy_update_freq (int): Frequency of policy updates (TD3 parameter)
            target_noise (float): Scale of noise added to target actions (TD3 parameter)
            noise_clip (float): Clipping range for target action noise (TD3 parameter)
        """
        self.num_agents = len(state_sizes)
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high
        self.policy_update_freq = policy_update_freq
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        
        self.total_state_size = sum(state_sizes)
        self.total_action_size = sum(action_sizes)
        
        self.agents = [DDPGAgent(
                            state_sizes[i], action_sizes[i], hidden_sizes,
                            actor_lr, critic_lr, tau,
                            centralized=True, total_state_size=self.total_state_size,
                            total_action_size=self.total_action_size,
                            action_low=action_low, action_high=action_high
                        ) for i in range(self.num_agents)]
        
        self._update_counter = 0

    def act(self, states, add_noise=True, noise_scale=0.1):
        """Get actions from all agents."""
        actions = [agent.act(state, add_noise, noise_scale) 
                   for agent, state in zip(self.agents, states)]
        return actions
        
    def act_target(self, states):
        """Get target actions from all agents."""
        return [agent.act_target(state) for agent, state in zip(self.agents, states)]
        
    def learn(self, experiences, agent_idx):
        """
        Update policy and value parameters for a specific agent using TD3 logic.
        """
        states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full = experiences
        current_agent = self.agents[agent_idx]
        
        agent_rewards = rewards[agent_idx]
        agent_dones = dones[agent_idx]

        # ---------------------------- update centralized critic ---------------------------- #
        with torch.no_grad():
            # --- Target Policy Smoothing ---
            # Get next actions from all agents' target networks
            next_actions_list = self.act_target(next_states)
            
            # Add noise to target actions
            noise = [torch.randn_like(a) * self.target_noise for a in next_actions_list]
            clipped_noise = [torch.clamp(n, -self.noise_clip, self.noise_clip) for n in noise]
            
            # Add clipped noise to the next actions
            noisy_next_actions_list = [torch.clamp(a + n, self.action_low, self.action_high) 
                                       for a, n in zip(next_actions_list, clipped_noise)]
            
            next_actions_full = torch.cat(noisy_next_actions_list, dim=1)
            
            # --- Clipped Double Q-Learning ---
            # Compute target Q-value from both critic target networks
            Q_targets_next1 = current_agent.critic_target(next_states_full, next_actions_full)
            Q_targets_next2 = current_agent.critic_target2(next_states_full, next_actions_full)
            
            # Take the minimum of the two target Q-values
            Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)
            
            # Compute final Q targets
            Q_targets = agent_rewards + (self.gamma * Q_targets_next * (1 - agent_dones))

        # Compute critic loss for both critics
        Q_expected1 = current_agent.critic(states_full, actions_full)
        Q_expected2 = current_agent.critic2(states_full, actions_full)
        
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets)
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets)
        critic_loss = critic_loss1 + critic_loss2

        # Update both critics
        current_agent.critic_optimizer.zero_grad()
        current_agent.critic_optimizer2.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.critic.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(current_agent.critic2.parameters(), 1.0)
        current_agent.critic_optimizer.step()
        current_agent.critic_optimizer2.step()
        
        # Increment the update counter for this agent
        self._update_counter += 1
        actor_loss = None # Initialize actor_loss to None

        # --- Delayed Policy Updates ---
        if self._update_counter % self.policy_update_freq == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred_list = []
            for i, agent in enumerate(self.agents):
                # Use current agent's actor, others are from experience
                if i == agent_idx:
                    actions_pred_list.append(current_agent.actor(states[i]))
                else:
                    actions_pred_list.append(actions[i].detach())
            
            actions_full_pred = torch.cat(actions_pred_list, dim=1)
            
            # Use the first critic for actor loss calculation (standard practice)
            actor_loss_tensor = -current_agent.critic(states_full, actions_full_pred).mean()
            actor_loss = actor_loss_tensor.item()

            # Update the actor
            current_agent.actor_optimizer.zero_grad()
            actor_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(current_agent.actor.parameters(), 0.5)
            current_agent.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            # Soft update all target networks for the current agent
            current_agent.soft_update(current_agent.actor_target, current_agent.actor)
            current_agent.soft_update(current_agent.critic_target, current_agent.critic)
            current_agent.soft_update(current_agent.critic_target2, current_agent.critic2)

        return critic_loss.item(), actor_loss if actor_loss is not None else 0.0

    def update_targets(self):
        """
        In MATD3, target updates are delayed and handled within the `learn` method
        for each agent individually. This global method is kept for API consistency
        but does nothing.
        """
        pass

    def save(self, path):
        """Save all agent models to a single file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        models_dict = {
            f'agent_{i}_actor': agent.actor.state_dict() for i, agent in enumerate(self.agents)
        }
        for i, agent in enumerate(self.agents):
            models_dict[f'agent_{i}_critic'] = agent.critic.state_dict()
            models_dict[f'agent_{i}_critic2'] = agent.critic2.state_dict()
        
        torch.save(models_dict, path)
        print(f"MATD3 models saved to {path}")
    
    def load(self, path):
        """Load all agent models from a single file."""
        if not os.path.exists(path):
            print(f"Warning: No model file found at {path}")
            return
            
        models_dict = torch.load(path)
        
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(models_dict[f'agent_{i}_actor'])
            agent.actor_target.load_state_dict(models_dict[f'agent_{i}_actor'])

            agent.critic.load_state_dict(models_dict[f'agent_{i}_critic'])
            agent.critic_target.load_state_dict(models_dict[f'agent_{i}_critic'])

            # Load the second critic if it exists in the file
            critic2_key = f'agent_{i}_critic2'
            if critic2_key in models_dict:
                agent.critic2.load_state_dict(models_dict[critic2_key])
                agent.critic_target2.load_state_dict(models_dict[critic2_key])
        
        print(f"All MATD3 models loaded from {path}")