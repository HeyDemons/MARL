import os
import numpy as np
import torch
import torch.nn.functional as F
from .td3_agent import TD3  # 导入新的TD3智能体
from .buffer import ReplayBuffer

class MATD3():
    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, action_bound, _chkpt_dir, _device='cpu', _model_timestamp=None):
        if _chkpt_dir is not None:
            os.makedirs(_chkpt_dir, exist_ok=True)

        self.device = _device
        self.model_timestamp = _model_timestamp
        self.agent_ids = list(dim_info.keys())
        
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())

        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = TD3(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device, action_bound[agent_id], chkpt_name=(agent_id + '_'), chkpt_dir=_chkpt_dir)
            self.buffers[agent_id] = ReplayBuffer(capacity, obs_dim, act_dim, self.device)
        
        self.dim_info = dim_info
        self.batch_size = batch_size
        self.total_it = 0 # 用于延迟策略更新的计数器

    def add(self, obs, action, reward, next_obs, done):
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)

    def sample(self, batch_size):
        total_num = len(self.buffers[self.agent_ids[0]])
        if total_num < batch_size:
            return None
        indices = np.random.choice(total_num, size=batch_size, replace=False)
        
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d = buffer.sample(indices)
            obs_batch[agent_id] = o
            act_batch[agent_id] = a
            rew_batch[agent_id] = r
            next_obs_batch[agent_id] = n_o
            done_batch[agent_id] = d
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch

    def select_action(self, obs):
        actions = {}
        with torch.no_grad():
            for agent_id, o in obs.items():
                o_tensor = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
                action, _ = self.agents[agent_id].action(o_tensor)
                actions[agent_id] = action.squeeze(0).cpu().numpy()
        return actions

    def learn(self, batch_size, gamma, policy_noise=0.2, noise_clip=0.5, policy_delay=2, tau=0.01):
        self.total_it += 1
        
        # 从所有智能体的buffer中采样
        sample_batches = self.sample(batch_size)
        if sample_batches is None:
            return # 样本不足，无法学习

        obs, act, reward, next_obs, done = sample_batches

        # 遍历每个智能体进行更新
        for agent_id, agent in self.agents.items():
            # --- 1. 计算目标Q值 ---
            with torch.no_grad():
                # 目标策略平滑：给目标动作加噪声
                next_act = {}
                for id, a in self.agents.items():
                    # 获取其他智能体的下一个动作
                    next_action_tensor, _ = a.target_action(next_obs[id])
                    
                    # 添加噪声
                    noise = (torch.randn_like(next_action_tensor) * policy_noise).clamp(-noise_clip, noise_clip)
                    
                    # 限制动作范围
                    min_action, max_action = a.actor.action_bound
                    noisy_next_action = (next_action_tensor + noise).clamp(min_action, max_action)
                    next_act[id] = noisy_next_action

                # 双评论家网络：计算两个目标Critic的值，并取较小者
                next_target_critic_1_value = agent.target_critic_1_value(list(next_obs.values()), list(next_act.values()))
                next_target_critic_2_value = agent.target_critic_2_value(list(next_obs.values()), list(next_act.values()))
                next_target_critic_value = torch.min(next_target_critic_1_value, next_target_critic_2_value)

                # 计算最终的目标值
                target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            # --- 2. 更新两个Critic网络 ---
            critic_1_value = agent.critic_1_value(list(obs.values()), list(act.values()))
            critic_2_value = agent.critic_2_value(list(obs.values()), list(act.values()))

            critic_1_loss = F.mse_loss(critic_1_value, target_value.detach())
            critic_2_loss = F.mse_loss(critic_2_value, target_value.detach())
            
            agent.update_critic(critic_1_loss, critic_2_loss)

            # --- 3. 延迟更新Actor和目标网络 ---
            if self.total_it % policy_delay == 0:
                # 获取当前智能体的新动作
                new_action, _ = agent.action(obs[agent_id])
                
                # 创建新的联合动作，替换掉当前智能体的旧动作
                new_act_batch = act.copy()
                new_act_batch[agent_id] = new_action

                # 计算Actor损失
                # 注意：这里只使用critic_1的输出来计算actor_loss是TD3的标准做法
                actor_loss = -agent.critic_1_value(list(obs.values()), list(new_act_batch.values())).mean()
                agent.update_actor(actor_loss)

                # --- 4. 软更新所有目标网络 ---
                self.update_target(tau)

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic_1, agent.target_critic_1)
            soft_update(agent.critic_2, agent.target_critic_2)
            
    def save_model(self):
        """Saves the models for each agent."""
        for agent_id, agent in self.agents.items():
            agent.actor.save_checkpoint(timestamp=True)
            agent.critic_1.save_checkpoint(timestamp=True)
            agent.critic_2.save_checkpoint(timestamp=True)
            # 同样需要保存目标网络的状态，以便在需要时恢复
            # 在您提供的代码中，DDPG的目标网络保存逻辑有些复杂，这里简化为只保存主网络
            # 如果需要完全恢复，也应保存目标网络

    def load_model(self):
        """Loads the models for each agent from a specific timestamp."""
        print(f"--- Loading models from timestamp {self.model_timestamp} ---")
        for agent_id, agent in self.agents.items():
            agent.actor.load_checkpoint(device=self.device, timestamp=self.model_timestamp)
            agent.critic_1.load_checkpoint(device=self.device, timestamp=self.model_timestamp)
            agent.critic_2.load_checkpoint(device=self.device, timestamp=self.model_timestamp)
            # 加载后，将主网络参数复制到目标网络
            agent.target_actor.load_state_dict(agent.actor.state_dict())
            agent.target_critic_1.load_state_dict(agent.critic_1.state_dict())
            agent.target_critic_2.load_state_dict(agent.critic_2.state_dict())