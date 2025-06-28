# 文件: matd3.py 
"""
Multi-Agent Twin-Delayed DDPG (MATD3) Implementation
"""
import os
import torch
import torch.nn.functional as F
from .ddpg import DDPGAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MATD3:
    """
    Multi-Agent Twin-Delayed DDPG (MATD3) implementation
    """
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64), 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3, 
                 action_low=-1.0, action_high=1.0,
                 policy_update_freq=2, target_noise=0.2, noise_clip=0.5,
                 shared_groups=None): # 新增参数
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
            shared_groups (dict, optional): Defines agent groups for parameter sharing. 
                                            Example: {'group1': [0, 1], 'group2': [2, 3]}. 
                                            Defaults to None (no sharing).
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
        
        if shared_groups:
            # --- 参数共享模式 ---
            self.agent_models = {}
            self.agent_map = [None] * self.num_agents
            for group_name, agent_indices in shared_groups.items():
                # 为每个组创建一个共享的 DDPGAgent 实例
                first_agent_idx = agent_indices[0]
                shared_agent = DDPGAgent(
                    state_sizes[first_agent_idx], action_sizes[first_agent_idx], hidden_sizes,
                    actor_lr, critic_lr, tau,
                    centralized=True, total_state_size=self.total_state_size,
                    total_action_size=self.total_action_size,
                    action_low=action_low, action_high=action_high
                )
                self.agent_models[group_name] = shared_agent
                # 将组内所有智能体索引映射到这个共享实例
                for agent_idx in agent_indices:
                    self.agent_map[agent_idx] = shared_agent
            self.agents = self.agent_map # 保持API一致性
        else:
            # --- 独立模式 (原始行为) ---
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
        # 此处代码无需修改，因为它迭代 self.agents，而 self.agents 已经被正确设置
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
        # 此处代码无需修改，self.agents[agent_idx] 会自动获取到正确的共享模型或独立模型
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
        
        # 根据是否共享参数来构建模型字典
        if hasattr(self, 'agent_models'): # 共享模式
            models_dict = {
                f'group_{name}_actor': model.actor.state_dict() for name, model in self.agent_models.items()
            }
            for name, model in self.agent_models.items():
                models_dict[f'group_{name}_critic'] = model.critic.state_dict()
                models_dict[f'group_{name}_critic2'] = model.critic2.state_dict()
        else: # 独立模式
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
            print(f"Error: Model file not found at {path}")
            return
            
        models_dict = torch.load(path, map_location=device)
        
        if hasattr(self, 'agent_models'): # 共享模式
            for name, model in self.agent_models.items():
                model.actor.load_state_dict(models_dict[f'group_{name}_actor'])
                model.critic.load_state_dict(models_dict[f'group_{name}_critic'])
                if f'group_{name}_critic2' in models_dict:
                    model.critic2.load_state_dict(models_dict[f'group_{name}_critic2'])
                # Hard update target networks
                model.hard_update(model.actor_target, model.actor)
                model.hard_update(model.critic_target, model.critic)
                model.hard_update(model.critic_target2, model.critic2)
        else: # 独立模式
            for i, agent in enumerate(self.agents):
                agent.actor.load_state_dict(models_dict[f'agent_{i}_actor'])
                agent.critic.load_state_dict(models_dict[f'agent_{i}_critic'])
                if f'agent_{i}_critic2' in models_dict:
                    agent.critic2.load_state_dict(models_dict[f'agent_{i}_critic2'])
                # Hard update target networks
                agent.hard_update(agent.actor_target, agent.actor)
                agent.hard_update(agent.critic_target, agent.critic)
                agent.hard_update(agent.critic_target2, agent.critic2)
        
        print(f"MATD3 models loaded from {path}")