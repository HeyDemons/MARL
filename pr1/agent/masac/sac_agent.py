from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
from .actor import Actor
from .critic import Critic

class SAC():
    def __init__(self, obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, alpha_lr, device, action_bound, chkpt_dir, chkpt_name, max_grad_norm=1e7):
        self.device = device
        
        # Actor Network
        self.actor = Actor(in_dim=obs_dim, out_dim=act_dim, action_bound=action_bound, chkpt_dir=chkpt_dir, chkpt_name=(chkpt_name + 'actor.pth')).to(device)

        # Critic Networks
        self.critic_1 = Critic(in_dim=global_obs_act_dim, out_dim=1, chkpt_dir=chkpt_dir, chkpt_name=(chkpt_name + 'critic1.pth')).to(device)
        self.critic_2 = Critic(in_dim=global_obs_act_dim, out_dim=1, chkpt_dir=chkpt_dir, chkpt_name=(chkpt_name + 'critic2.pth')).to(device)
        
        # Target Critic Networks
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=critic_lr)

        # Automatic Temperature (Alpha) Tuning
        self.target_entropy = -torch.prod(torch.Tensor(act_dim).to(device)).item()*1.5
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        
        self.max_grad_norm = max_grad_norm
        
    def get_action(self, obs, model_out = False):
        action, log_prob = self.actor(obs)
        if model_out:
            return action, log_prob
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def update_critic(self, critic_loss_1, critic_loss_2):
        # 更新 Critic 1
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        self.critic_1_optimizer.step()

        # 更新 Critic 2
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_2_optimizer.step()

    def update_actor_and_alpha(self, actor_loss, alpha_loss):
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # 保留计算图
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # 更新 Alpha
        if self.alpha_optimizer:  # 假设你有 alpha 的优化器
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()  # 这里不需要 retain_graph，因为是最后一次 backward
            self.alpha_optimizer.step()