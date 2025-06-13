import os
import numpy as np
import torch
import torch.nn.functional as F
from .sac_agent import SAC
from ..maddpg.buffer import ReplayBuffer

class MASAC():
    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, alpha_lr, action_bound, _chkpt_dir, _device='cpu', _model_timestamp=None):
        if _chkpt_dir is not None:
            os.makedirs(_chkpt_dir, exist_ok=True)

        self.device = _device
        self.model_timestamp = _model_timestamp
        self.agent_ids = list(dim_info.keys())
        
        global_obs_dim = sum(dims[0] for dims in dim_info.values())
        global_act_dim = sum(dims[1] for dims in dim_info.values())
        global_obs_act_dim = global_obs_dim + global_act_dim

        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = SAC(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, alpha_lr, self.device, action_bound[agent_id], chkpt_name=(agent_id + '_'), chkpt_dir=_chkpt_dir)
            self.buffers[agent_id] = ReplayBuffer(capacity, obs_dim, act_dim, self.device)
        
        self.dim_info = dim_info
        self.batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):  #返回值为True or False, 判断a是否为int类型，是，返回True。
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from each agent's buffer.
        """
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
                action, _ = self.agents[agent_id].get_action(o_tensor)
                actions[agent_id] = action.squeeze(0)
        return actions

# masac_agent.py

    def learn(self, batch_size, gamma, tau=0.005):
        sample_batches = self.sample(batch_size)
        if sample_batches is None:
            return # Not enough samples to learn
        
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = sample_batches
            
        global_obs = torch.cat(list(obs_batch.values()), dim=1)
        global_act = torch.cat(list(act_batch.values()), dim=1)
        global_next_obs = torch.cat(list(next_obs_batch.values()), dim=1)

        # --- Start of Corrected Logic ---

        # Get next actions from all agents for constructing the global_next_act
        with torch.no_grad():
            next_actions, next_log_probs = {}, {}
            for agent_id, agent in self.agents.items():
                # Each agent uses its own actor to determine its next action from its own observation
                next_actions[agent_id], next_log_probs[agent_id] = agent.actor(next_obs_batch[agent_id])
            global_next_act = torch.cat(list(next_actions.values()), dim=1)
            sum_next_log_probs = torch.cat(list(next_log_probs.values()), dim=1).sum(dim=1, keepdim=True)

        # Loop through each agent to perform its individual update
        for agent_id, agent in self.agents.items():
            
            # 1. CALCULATE THE BELLMAN TARGET FOR THIS SPECIFIC AGENT
            with torch.no_grad():
                # FIX: Use this agent's OWN target critic networks
                q1_target = agent.target_critic_1(torch.cat([global_next_obs, global_next_act], 1))
                q2_target = agent.target_critic_2(torch.cat([global_next_obs, global_next_act], 1))
                q_target = torch.min(q1_target, q2_target)
                agent_next_log_probs = next_log_probs[agent_id]
                # The entropy term is based on the sum of all agents' action log-probabilities
                # FIX: Use this agent's OWN alpha
                q_target_entropy = q_target - agent.alpha * agent_next_log_probs
                
                # FIX: Use this agent's OWN reward and done signal
                reward_for_update = rew_batch[agent_id].unsqueeze(1)
                done_for_update = done_batch[agent_id].unsqueeze(1)

                # This is the correct Bellman target for the current agent
                bellman_target = reward_for_update + gamma * (1 - done_for_update) * q_target_entropy

            # 2. UPDATE THIS AGENT'S CRITICS
            current_q1 = agent.critic_1(torch.cat([global_obs, global_act], 1))
            current_q2 = agent.critic_2(torch.cat([global_obs, global_act], 1))
            critic_loss_1 = F.mse_loss(current_q1, bellman_target)
            critic_loss_2 = F.mse_loss(current_q2, bellman_target)
            agent.update_critic(critic_loss_1, critic_loss_2)
            
            # 3. UPDATE THIS AGENT'S ACTOR AND ALPHA
            # We need to re-calculate actions and log_probs to keep them in the computation graph
            new_actions, log_probs = agent.actor(obs_batch[agent_id])
            
            # Create a temporary action batch for the new global action
            temp_act_batch = act_batch.copy()
            temp_act_batch[agent_id] = new_actions
            new_global_act = torch.cat(list(temp_act_batch.values()), dim=1)
            
            # Calculate the actor loss
            q1_new = agent.critic_1(torch.cat([global_obs, new_global_act], 1))
            q2_new = agent.critic_2(torch.cat([global_obs, new_global_act], 1))
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (agent.alpha * log_probs - q_new).mean()
            
            # Calculate the alpha loss
            alpha_loss = -(agent.log_alpha * (log_probs + agent.target_entropy).detach()).mean()

            agent.update_actor_and_alpha(actor_loss, alpha_loss)

        # The target network update happens for all agents after their learning steps
        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(from_net, to_net):
            for from_p, to_p in zip(from_net.parameters(), to_net.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.critic_1, agent.target_critic_1)
            soft_update(agent.critic_2, agent.target_critic_2)
            
    def save_model(self):
        """Saves the models for each agent."""
        for agent_id, agent in self.agents.items():
            agent.actor.save_checkpoint(timestamp=True)
            agent.critic_1.save_checkpoint(timestamp=True)
            agent.critic_2.save_checkpoint(timestamp=True)
            agent.target_critic_1.save_checkpoint(is_target=True, timestamp=True)
            agent.target_critic_2.save_checkpoint(is_target=True, timestamp=True)

    def load_model(self):
        """Loads the models for each agent from a specific timestamp."""
        print(f"--- Loading models from timestamp {self.model_timestamp} ---")
        for agent_id, agent in self.agents.items():
            agent.actor.load_checkpoint(device=self.device, timestamp=self.model_timestamp)
            agent.critic_1.load_checkpoint(device=self.device, timestamp=self.model_timestamp)
            agent.critic_2.load_checkpoint(device=self.device, timestamp=self.model_timestamp)
            agent.target_critic_1.load_checkpoint(device=self.device, is_target=True, timestamp=self.model_timestamp)
            agent.target_critic_2.load_checkpoint(device=self.device, is_target=True, timestamp=self.model_timestamp)