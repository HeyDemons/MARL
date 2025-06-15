# qmix_mpe.py (已更新为支持 GPU)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os # <--- 新增：导入 os 模块


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class DRQN(nn.Module):
    # --- 此网络结构部分无需修改 ---
    def __init__(self, input_shape, args):
        super(DRQN, self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        orthogonal_init(self.fc1, gain=nn.init.calculate_gain('relu'))
        orthogonal_init(self.fc2, gain=0.01)
    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class QmixNet(nn.Module):
    # --- 此网络结构部分无需修改 ---
    def __init__(self, args):
        super(QmixNet, self).__init__()
        self.args = args
        
        self.hyper_w1 = nn.Linear(args.state_dim, args.N * args.qmix_hidden_dim)
        self.hyper_b1 = nn.Linear(args.state_dim, args.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(args.state_dim, args.qmix_hidden_dim * 1)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_dim, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1))
        

    def forward(self, q_values, states):
        batch_size = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.N)
        states = states.reshape(-1, self.args.state_dim)

        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.args.N, self.args.qmix_hidden_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.args.qmix_hidden_dim)
        
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(batch_size, -1, 1)
        
        return q_total

class QMIX_MPE:
    def __init__(self, args, device): # <--- 修改：接收 device
        self.args = args
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.device = device # <--- 新增：存储 device

        # DRQN 网络 (评估网络和目标网络)
        self.eval_drqn_net = DRQN(self.obs_dim, args).to(self.device) # <--- 修改：上载到 device
        self.target_drqn_net = DRQN(self.obs_dim, args).to(self.device) # <--- 修改：上载到 device
        
        # Qmix 网络 (评估网络和目标网络)
        self.eval_qmix_net = QmixNet(args).to(self.device) # <--- 修改：上载到 device
        self.target_qmix_net = QmixNet(args).to(self.device) # <--- 修改：上载到 device

        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.drqn_params = list(self.eval_drqn_net.parameters())
        self.qmix_params = list(self.eval_qmix_net.parameters())
        self.optimizer = torch.optim.Adam(self.drqn_params + self.qmix_params, lr=self.args.lr)

        self.train_step = 0

    def choose_action(self, obs_n, hidden_state, epsilon, evaluate=False):
        # <--- 修改：将输入数据转换为 Tensor 并上载到 device
        obs_n = torch.tensor(obs_n, dtype=torch.float32).to(self.device)
        hidden_state = torch.tensor(hidden_state, dtype=torch.float32).to(self.device)
        actions = np.zeros(self.N)
        
        q_values, new_hidden_state = self.eval_drqn_net(obs_n, hidden_state)

        for i in range(self.N):
            if np.random.rand() < epsilon and not evaluate:
                actions[i] = np.random.randint(0, self.action_dim)
            else:
                actions[i] = q_values[i].argmax().item()

        # <--- 修改：将 device 上的 hidden_state 转换回 numpy
        return actions.astype(int), new_hidden_state.detach().cpu().numpy()
        
    def update_target_net(self):
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def train(self, batch):
        self.train_step += 1
        
        obs_n = torch.tensor(batch['obs_n'], dtype=torch.float32).to(self.device)
        s = torch.tensor(batch['s'], dtype=torch.float32).to(self.device)
        actions_n = torch.tensor(batch['a_n'], dtype=torch.long).to(self.device)
        r_n = torch.tensor(batch['r_n'], dtype=torch.float32).to(self.device)
        obs_next_n = torch.tensor(batch['obs_next_n'], dtype=torch.float32).to(self.device)
        s_next = torch.tensor(batch['s_next'], dtype=torch.float32).to(self.device)
        done_n = torch.tensor(batch['done_n'], dtype=torch.float32).to(self.device)
        actual_len = torch.tensor(batch['actual_len'], dtype=torch.long).to(self.device) # 新增：获取实际长度
        
        batch_size = obs_n.shape[0]
        episode_len = obs_n.shape[1] # 这是 episode_limit
        
        # 新增：创建 mask
        # mask 的形状应为 (batch_size, episode_len, 1)
        # 对于每个 episode，在 actual_len 之前的步为 1，之后为 0
        max_actual_len = actual_len.max().item() # 实际需要的序列长度
        mask_indices = torch.arange(episode_len, device=self.device).unsqueeze(0).repeat(batch_size, 1) # (batch_size, episode_len)
        mask = (mask_indices < actual_len).float().unsqueeze(-1) # (batch_size, episode_len, 1)
        
        q_evals, h_eval_list_unused = [], [] 
        hidden_eval = torch.zeros(batch_size, self.N, self.args.rnn_hidden_dim).to(self.device)
        
        # 只循环到 max_actual_len 即可，因为 mask 之外的 Q 值不影响损失
        # 但为了保持 q_evals 的形状为 (batch_size, episode_len, N, action_dim) 以便后续 gather
        # 还是循环到 episode_len，mask 会处理掉 padding 部分的损失
        for t in range(episode_len):
            obs_t = obs_n[:, t, :, :]
            obs_t_reshaped = obs_t.reshape(-1, self.obs_dim)
            h_reshaped = hidden_eval.reshape(-1, self.args.rnn_hidden_dim)
            
            q_t, h_out = self.eval_drqn_net(obs_t_reshaped, h_reshaped)
            q_evals.append(q_t.reshape(batch_size, self.N, -1))
            hidden_eval = h_out.reshape(batch_size, self.N, -1) 
            
        q_evals = torch.stack(q_evals, dim=1)
        chosen_action_qvals = torch.gather(q_evals, dim=3, index=actions_n.unsqueeze(3)).squeeze(3)
        
        q_total_eval = self.eval_qmix_net(chosen_action_qvals, s)

        q_targets_list = [] 
        hidden_target_rnn_input = torch.zeros(batch_size, self.N, self.args.rnn_hidden_dim).to(self.device)
        
        for t in range(episode_len): # 同上，循环到 episode_len
            obs_next_t = obs_next_n[:, t, :, :]
            obs_next_t_reshaped = obs_next_t.reshape(-1, self.obs_dim)
            h_target_reshaped = hidden_target_rnn_input.reshape(-1, self.args.rnn_hidden_dim)

            q_target_t, h_target_out = self.target_drqn_net(obs_next_t_reshaped, h_target_reshaped)
            q_targets_list.append(q_target_t.reshape(batch_size, self.N, -1))
            hidden_target_rnn_input = h_target_out.reshape(batch_size, self.N, -1) 
            
        q_targets_all_actions = torch.stack(q_targets_list, dim=1) 
        
        q_evals_next_list = []
        hidden_eval_for_next_obs = torch.zeros(batch_size, self.N, self.args.rnn_hidden_dim).to(self.device)

        for t in range(episode_len): # 同上，循环到 episode_len
            obs_next_t = obs_next_n[:, t, :, :]
            obs_next_t_reshaped = obs_next_t.reshape(-1, self.obs_dim)
            h_in_eval_next_obs = hidden_eval_for_next_obs.reshape(-1, self.args.rnn_hidden_dim)
            
            q_values_next_t, h_out_eval_next_obs = self.eval_drqn_net(obs_next_t_reshaped, h_in_eval_next_obs)
            q_evals_next_list.append(q_values_next_t.reshape(batch_size, self.N, -1))
            hidden_eval_for_next_obs = h_out_eval_next_obs.reshape(batch_size, self.N, -1)

        q_evals_next = torch.stack(q_evals_next_list, dim=1) 
        
        best_actions = q_evals_next.argmax(dim=3, keepdim=True) 
        q_target_next = torch.gather(q_targets_all_actions, 3, best_actions).squeeze(3)
        
        q_total_target = self.target_qmix_net(q_target_next, s_next)
        reward = r_n.sum(dim=2, keepdim=True) 
        team_done = done_n.all(dim=2, keepdim=True).float()

        target = reward + self.args.gamma * q_total_target * (1 - team_done) 
        
        td_error = q_total_eval - target.detach()
        
        # 修改：应用 mask 计算 loss
        masked_td_error = td_error * mask
        # 只有当 mask.sum() > 0 时才进行除法，避免 NaN
        if mask.sum() > 0:
            loss = (masked_td_error ** 2).sum() / mask.sum()
        else:
            loss = torch.tensor(0.0).to(self.device) # 或者其他处理方式
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.drqn_params + self.qmix_params, self.args.grad_norm_clip)
        self.optimizer.step()
        
        if self.train_step > 0 and self.train_step % self.args.target_update_freq == 0:
            self.update_target_net()
            
    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.eval_drqn_net.state_dict(), f"./model/QMIX_drqn_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps/1000)}k.pth")
        torch.save(self.eval_qmix_net.state_dict(), f"./model/QMIX_qmix_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps/1000)}k.pth")
    
    def load_model(self, env_name, number, seed, total_steps_k):
        """
        Load pre-trained model weights.
        total_steps_k is the step number in thousands (e.g., 3000 for 3,000,000 steps).
        """
        drqn_path = f"./model/QMIX_drqn_env_{env_name}_number_{number}_seed_{seed}_step_{total_steps_k}k.pth"
        qmix_path = f"./model/QMIX_qmix_env_{env_name}_number_{number}_seed_{seed}_step_{total_steps_k}k.pth"
        
        if os.path.exists(drqn_path) and os.path.exists(qmix_path):
            self.eval_drqn_net.load_state_dict(torch.load(drqn_path, map_location=self.device))
            self.eval_qmix_net.load_state_dict(torch.load(qmix_path, map_location=self.device))
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict()) # Also update target nets
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
            print(f"Successfully loaded QMIX models from: {drqn_path} and {qmix_path}")
        else:
            print(f"Error: Could not find model files at {drqn_path} or {qmix_path}")
            raise FileNotFoundError("QMIX model files not found.")