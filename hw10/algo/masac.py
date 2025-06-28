# 文件: masac.py 
"""
Multi-Agent Soft Actor-Critic (MASAC) Implementation
This version uses the existing ApproxActor from networks.py without modification.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

# 直接从你现有的 networks.py 导入所需的模块
from .networks import SACActor
# 我们可以复用 DDPGAgent 作为网络和优化器的容器
from .ddpg import DDPGAgent 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MASAC:
    """Multi-Agent Soft Actor-Critic (MASAC)"""

    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64),
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 action_low=-1.0, action_high=1.0,
                 target_entropy=None):
        """
        Initialize a MASAC agent.
        """
        self.num_agents = len(state_sizes)
        self.gamma = gamma
        self.tau = tau
        
        self.total_state_size = sum(state_sizes)
        self.total_action_size = sum(action_sizes)

        # 创建智能体列表
        self.agents = []
        for i in range(self.num_agents):
            # 先创建一个 DDPGAgent 实例作为容器
            agent = DDPGAgent(
                state_sizes[i], action_sizes[i], hidden_sizes,
                actor_lr, critic_lr, tau,
                centralized=True, total_state_size=self.total_state_size,
                total_action_size=self.total_action_size,
                action_low=action_low, action_high=action_high
            )
            
            # **关键步骤**: 将容器中默认的确定性 Actor 替换为 SACActor 的实例
            agent.actor = SACActor(
                state_size=state_sizes[i],
                action_size=action_sizes[i],
                hidden_sizes=hidden_sizes,
                action_low=action_low,
                action_high=action_high
            ).to(device)
            agent.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=actor_lr)
            
            self.agents.append(agent)
            
        # 可学习的 alpha (温度)
        if target_entropy is None:
            # target_entropy 的一个常用启发式设置是动作空间维度的相反数
            self.target_entropy = -float(action_sizes[0]) 
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(np.log(0.2),dtype = torch.float32, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def act(self, states, deterministic=False):
        """从所有智能体的策略中获取动作。"""
        actions = []
        with torch.no_grad():
            for i, state in enumerate(states):
                state_tensor = torch.from_numpy(state).float().to(device)
                # 调用 ApproxActor 的 sample 方法, 我们只需要动作
                action, _, _ = self.agents[i].actor.sample(state_tensor, deterministic=deterministic)
                actions.append(action.cpu().numpy())
        return actions

    def learn(self, experiences, agent_idx):
        """为指定智能体更新策略、Critic 和 alpha。"""
        states, actions, rewards, next_states, dones, states_full, next_states_full, _ = experiences
        current_agent = self.agents[agent_idx]
        
        agent_rewards = rewards[agent_idx]
        agent_dones = dones[agent_idx]

        # ---------------------------- 更新 Critic ---------------------------- #
        with torch.no_grad():
            # 为所有智能体获取下一个动作及其对数概率
            next_actions_list = []
            next_log_probs_list = []
            for i, next_state in enumerate(next_states):
                # 调用 ApproxActor 的 sample 方法
                next_action, next_log_prob, _ = self.agents[i].actor.sample(next_state)
                next_actions_list.append(next_action)
                next_log_probs_list.append(next_log_prob)

            next_actions_full = torch.cat(next_actions_list, dim=1)
            agent_next_log_prob = next_log_probs_list[agent_idx]

            # 从两个目标 Critic 计算 Q 目标值
            q_next_target1 = current_agent.critic_target(next_states_full, next_actions_full)
            q_next_target2 = current_agent.critic_target2(next_states_full, next_actions_full)
            q_next_target = torch.min(q_next_target1, q_next_target2)

            # 在 Q 目标上减去熵项 (SAC的核心)
            q_target = agent_rewards + self.gamma * (1 - agent_dones) * \
                       (q_next_target - self.alpha * agent_next_log_prob)

        # 计算 Critic 损失
        actions_full = torch.cat(actions, dim=1)
        q1 = current_agent.critic(states_full, actions_full)
        q2 = current_agent.critic2(states_full, actions_full)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # 更新两个 Critic 网络
        current_agent.critic_optimizer.zero_grad()
        current_agent.critic_optimizer2.zero_grad()
        critic_loss.backward()
        current_agent.critic_optimizer.step()
        current_agent.critic_optimizer2.step()

        # --- 更新 Actor 和 Alpha ---
        # 为当前智能体采样动作和对数概率
        agent_actions_pred, agent_log_probs, _ = current_agent.actor.sample(states[agent_idx])
        
        actions_pred_list = list(actions)
        actions_pred_list[agent_idx] = agent_actions_pred
        actions_full_pred = torch.cat(actions_pred_list, dim=1)

        # 计算 Actor 损失
        q1_pred = current_agent.critic(states_full, actions_full_pred)
        q2_pred = current_agent.critic2(states_full, actions_full_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.alpha.detach() * agent_log_probs - q_pred).mean()

        # 更新 Actor
        current_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        current_agent.actor_optimizer.step()
        
        # 更新 Alpha
        alpha_loss = -(self.log_alpha * (agent_log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.alpha.item()

    def update_targets(self):
        """软更新所有目标网络。"""
        for agent in self.agents:
            agent.soft_update(agent.critic_target, agent.critic)
            agent.soft_update(agent.critic_target2, agent.critic2)

    def save(self, path):
        """保存所有智能体模型和alpha。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        models_dict = {
            f'agent_{i}_actor': agent.actor.state_dict() for i, agent in enumerate(self.agents)
        }
        for i, agent in enumerate(self.agents):
            models_dict[f'agent_{i}_critic'] = agent.critic.state_dict()
            models_dict[f'agent_{i}_critic2'] = agent.critic2.state_dict()
        
        models_dict['log_alpha'] = self.log_alpha
        torch.save(models_dict, path)
        print(f"MASAC models saved to {path}")
    
    def load(self, path):
        """加载所有智能体模型和alpha。"""
        if not os.path.exists(path):
            print(f"Warning: No model file found at {path}")
            return
            
        models_dict = torch.load(path)
        
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(models_dict[f'agent_{i}_actor'])
            agent.critic.load_state_dict(models_dict[f'agent_{i}_critic'])
            agent.critic_target.load_state_dict(models_dict[f'agent_{i}_critic'])
            
            critic2_key = f'agent_{i}_critic2'
            if critic2_key in models_dict:
                agent.critic2.load_state_dict(models_dict[critic2_key])
                agent.critic_target2.load_state_dict(models_dict[critic2_key])
        
        if 'log_alpha' in models_dict:
            self.log_alpha = models_dict['log_alpha']
            self.alpha = self.log_alpha.exp()
        
        print(f"All MASAC models loaded from {path}")