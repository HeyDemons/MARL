import os
import numpy as np
import torch 
import torch.nn.functional as F
from torch.optim import Adam
from .actor import ActorPPO
from .critic import CriticPPO
from .on_policy_buffer import OnPolicyBuffer

class MAPPO():
    def __init__(self, dim_info, buffer_capacity, mini_batch_size, actor_lr, critic_lr,
                  _chkpt_dir, _device='cpu', _model_timestamp=None,gamma=0.95, gae_lambda=0.95, clip_coef=0.2, entropy_coef=0.01):
        
        if _chkpt_dir is not None:
            os.makedirs(_chkpt_dir, exist_ok=True)

        self.device = _device
        self.model_timestamp = _model_timestamp
        self.dim_info = dim_info
        
        # Hyperparameters for PPO
        self.mini_batch_size = mini_batch_size
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef

        # The critic takes the global observation as input
        global_obs_dim = sum(val[0] for val in dim_info.values())

        self.agents = {}
        self.buffers = {}
        self.optimizers = {}

        for agent_id, (obs_dim, act_dim) in dim_info.items():
            actor = ActorPPO(
                chkpt_name=f"{agent_id}_actor_ppo.pth",
                chkpt_dir=_chkpt_dir,
                in_dim=obs_dim,
                out_dim=act_dim
            ).to(self.device)
            
            # The critic is centralized and shared among agents in this setup
            critic = CriticPPO(
                chkpt_name=f"shared_critic_ppo.pth", # Shared critic
                chkpt_dir=_chkpt_dir,
                in_dim=global_obs_dim
            ).to(self.device)

            actor_optimizer = Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
            critic_optimizer = Adam(critic.parameters(), lr=critic_lr, eps=1e-5)

            self.agents[agent_id] = {'actor': actor, 'critic': critic} # All agents refer to the same critic instance
            self.optimizers[agent_id] = {'actor': actor_optimizer, 'critic': critic_optimizer}

            self.buffers[agent_id] = OnPolicyBuffer(buffer_capacity, obs_dim, act_dim, self.device, gamma, gae_lambda)

    def select_action(self, obs):
        actions = {}
        log_probs = {}
        values = {}
        
        # Create global state tensor for the critic
        # Assuming obs is a dictionary of numpy arrays
        global_obs_list = [torch.from_numpy(o).float().to(self.device) for o in obs.values()]
        global_obs = torch.cat(global_obs_list, dim=0).unsqueeze(0)

        for agent_id, o in obs.items():
            o_tensor = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                # Get action from actor
                dist = self.agents[agent_id]['actor'](o_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # Get value from the shared critic
                value = self.agents[agent_id]['critic'](global_obs)
            
            actions[agent_id] = action.cpu().numpy().flatten()
            log_probs[agent_id] = log_prob.cpu().numpy().flatten()
            values[agent_id] = value.cpu().numpy().flatten()

        return actions, log_probs, values

    def add(self, obs, action, reward, done, log_probs, values):
        for agent_id in obs.keys():
            act_dim = self.dim_info[agent_id][1]
            a_one_hot = np.zeros(act_dim)
            # action from env.step is often an integer for discrete spaces
            a_one_hot[action[agent_id]] = 1.0

            self.buffers[agent_id].add(obs[agent_id], a_one_hot, reward[agent_id], 
                                       done[agent_id], log_probs[agent_id], values[agent_id])

    def learn(self):
        # 1. Compute advantages for all buffers first
        for agent_id, buffer in self.buffers.items():
            if len(buffer) == 0: continue
            
            last_done = buffer.dones[buffer._index-1]
            
            # To get the last value, we need the final observation
            final_obs_list = [torch.from_numpy(b.obs[b._index-1]).float().to(self.device) for b in self.buffers.values()]
            final_global_obs = torch.cat(final_obs_list, dim=0).unsqueeze(0)
            
            with torch.no_grad():
                last_value = self.agents[agent_id]['critic'](final_global_obs).cpu().numpy().flatten()
            
            buffer.compute_advantages_and_returns(last_value, last_done)
        
        # 2. Perform PPO updates for several epochs
        for _ in range(self.n_epochs):
            # Concatenate all agent data for critic update
            full_obs_data = {agent_id: buffer.get_data() for agent_id, buffer in self.buffers.items() if buffer.get_data() is not None}
            if not full_obs_data: continue # Skip if no data
                
            global_obs_tensors = [data[0] for data in full_obs_data.values()]
            global_obs_batch = torch.cat(global_obs_tensors, dim=1)
            
            # The returns for the critic are the same regardless of the agent
            critic_returns = next(iter(full_obs_data.values()))[4]

            for agent_id, agent_components in self.agents.items():
                if agent_id not in full_obs_data: continue

                actor = agent_components['actor']
                critic = agent_components['critic'] # Shared critic
                actor_optim = self.optimizers[agent_id]['actor']
                critic_optim = self.optimizers[agent_id]['critic'] # Shared optimizer

                obs_batch, act_batch, old_log_probs, advantages, returns = full_obs_data[agent_id]
                
                # Mini-batch updates
                batch_size = len(obs_batch)
                indices = np.arange(batch_size)
                np.random.shuffle(indices)

                for start in range(0, batch_size, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    batch_indices = indices[start:end]

                    # Actor Loss
                    dist = actor(obs_batch[batch_indices])
                    # Convert one-hot actions back to indices to get log_prob
                    action_indices = torch.argmax(act_batch[batch_indices], dim=1)
                    new_log_probs = dist.log_prob(action_indices)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                    
                    surr1 = ratio * advantages[batch_indices]
                    surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages[batch_indices]
                    
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                    
                    actor_optim.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optim.step()

                # Update the shared critic once per agent update cycle (or could be done separately)
                new_values = critic(global_obs_batch).squeeze()
                critic_loss = F.mse_loss(new_values, critic_returns)
                
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optim.step()

        # 3. Clear all buffers after the learning phase is complete
        for buffer in self.buffers.values():
            buffer.clear()

    def save_model(self):
        for agent_id, components in self.agents.items():
            components['actor'].save_checkpoint(timestamp=True)
        # Save the shared critic only once
        first_agent_id = next(iter(self.agents))
        self.agents[first_agent_id]['critic'].save_checkpoint(timestamp=True)


    def load_model(self):
        for agent_id, components in self.agents.items():
            components['actor'].load_checkpoint(device=self.device, timestamp=self.model_timestamp)
        # Load the shared critic only once
        first_agent_id = next(iter(self.agents))
        self.agents[first_agent_id]['critic'].load_checkpoint(device=self.device, timestamp=self.model_timestamp)