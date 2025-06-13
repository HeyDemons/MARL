import os
from copy import deepcopy
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from .actor import Actor
from .critic import Critic

class TD3():
    def __init__(self, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, device, action_bound, chkpt_dir, chkpt_name):
        self.device = device
        
        # Actor Network
        self.actor = Actor(in_dim=obs_dim, out_dim=act_dim, action_bound=action_bound, chkpt_dir=chkpt_dir, chkpt_name=(chkpt_name + 'actor.pth')).to(device)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Networks (TD3 uses two critics)
        self.critic_1 = Critic(in_dim=global_obs_act_dim, out_dim=1, chkpt_dir=chkpt_dir, chkpt_name=(chkpt_name + 'critic1.pth')).to(device)
        self.critic_2 = Critic(in_dim=global_obs_act_dim, out_dim=1, chkpt_dir=chkpt_dir, chkpt_name=(chkpt_name + 'critic2.pth')).to(device)
        
        # Target Critic Networks
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Optimizers for both critics
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=critic_lr)

    def action(self, obs, model_out=False):
        action, logi = self.actor(obs)
        return action, logi

    def target_action(self, obs):
        action, logi = self.target_actor(obs)
        return action, logi

    def critic_1_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic_1(x).squeeze(1)

    def critic_2_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic_2(x).squeeze(1)

    def target_critic_1_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic_1(x).squeeze(1)

    def target_critic_2_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic_2(x).squeeze(1)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss_1, loss_2):
        # Update Critic 1
        self.critic_1_optimizer.zero_grad()
        loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
        self.critic_1_optimizer.step()

        # Update Critic 2
        self.critic_2_optimizer.zero_grad()
        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_2_optimizer.step()