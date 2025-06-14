import torch
import torch.nn as nn
from torch.distributions import Normal
import os
from datetime import datetime

class Actor(nn.Module):
    def __init__(self,chkpt_name,  chkpt_dir, in_dim, out_dim, action_bound, hidden_dim = 128, non_linear = nn.ReLU()):
        super(Actor, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_name = chkpt_name
        self.action_bound = action_bound

        # SAC actor 输出高斯分布的均值和对数标准差
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
        )
        self.mean_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = nn.Linear(hidden_dim, out_dim)

        self.apply(self.init)

    @staticmethod
    def init(m):
        '''init patameters of the module'''
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = gain)  #使用了 Xavier 均匀分布初始化（也叫 Glorot 初始化）
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # 限制 log_std 的范围以保证稳定性
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        # 采样动作并进行 Tanh 压缩
        # 这是重参数化技巧
        unsquashed_action = dist.rsample()
        squashed_action = torch.tanh(unsquashed_action)
        
        # 将动作缩放到环境的动作空间
        action_range = torch.tensor((self.action_bound[1] - self.action_bound[0]) / 2, device=x.device)
        action_bias = torch.tensor((self.action_bound[1] + self.action_bound[0]) / 2, device=x.device)
        final_action = squashed_action * action_range + action_bias

        # 计算对数概率，并对 Tanh 压缩进行修正
        # 公式参考 SAC 原论文附录 C
        log_prob = dist.log_prob(unsquashed_action)
        log_prob -= torch.log(action_range * (1 - squashed_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return final_action, log_prob

    def save_checkpoint(self, is_target=False, timestamp = False):
        if timestamp is True:
             current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
             save_dir = os.path.join(self.chkpt_dir, current_timestamp)
        else:
            save_dir = self.chkpt_dir
        
        os.makedirs(save_dir, exist_ok=True)
        self.chkpt_file = os.path.join(save_dir, self.chkpt_name)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device = 'cpu', timestamp = None):
        if timestamp and isinstance(timestamp, str):
            load_dir = os.path.join(self.chkpt_dir, timestamp)
        else:
            load_dir = self.chkpt_dir
    
        self.chkpt_file = os.path.join(load_dir, self.chkpt_name)
        if not os.path.exists(self.chkpt_file):
            print(f"Warning: Could not find model file: {self.chkpt_file}")
            return
        self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))