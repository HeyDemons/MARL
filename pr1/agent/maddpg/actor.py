import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime

class Actor(nn.Module):
    def __init__(self,chkpt_name,  chkpt_dir, in_dim, out_dim, action_bound, hidden_dim = 64, non_linear = nn.ReLU()):
        super(Actor, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_name = chkpt_name

        # different ,为什么要保持这两个信息？
        self.out_dim = out_dim
        self.action_bound = action_bound

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        )
        # 初始化网络参数
        self.net_init(self.net[0])
        self.net_init(self.net[2])
        self.final_net_init(self.net[4], -3e-3, 3e-3)


    def net_init(self, layer):
        fan_in = layer.weight.data.size(0)
        limit = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(layer.weight, -limit, limit)
        nn.init.uniform_(layer.bias, -limit, limit)

    def final_net_init(self, layer, low, high):
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, low, high)
            nn.init.uniform_(layer.bias, low, high)
    
    def forward(self, x):
        x = self.net(x)
        a_min = self.action_bound[0]
        a_max = self.action_bound[1]
        # 将输出限制在指定范围内
        action = torch.tanh(x)
        return action

    def save_checkpoint(self, is_target=False, timestamp = False):
        # 使用时间戳保存功能
        if timestamp is True:
             # 使用时间戳创建新文件夹
             current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
             save_dir = os.path.join(self.chkpt_dir, current_timestamp)
        else:
            # 直接保存在主目录下，不使用时间戳
            save_dir = self.chkpt_dir
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建保存路径
        self.chkpt_file = os.path.join(save_dir, self.chkpt_name)

        if is_target:
            target_chkpt_name = self.chkpt_file.replace('actor', 'target_actor')
            os.makedirs(os.path.dirname(target_chkpt_name), exist_ok=True)
            torch.save(self.state_dict(), target_chkpt_name)
        else:
            os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
            torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device = 'cpu', is_target = False, timestamp = None): # 默认加载target
        if timestamp and isinstance(timestamp, str):
            # 如果提供了有效的时间戳字符串，从对应文件夹加载
            load_dir = os.path.join(self.chkpt_dir, timestamp)
        else:
            # 否则从主目录加载
            load_dir = self.chkpt_dir
    
        # 使用os.path.join确保路径分隔符的一致性
        self.chkpt_file = os.path.join(load_dir, self.chkpt_name)
    
        if is_target:
            target_chkpt_name = self.chkpt_file.replace('actor', 'target_actor')
            # 确保路径存在
            if not os.path.exists(target_chkpt_name):
                print(f"警告: 找不到目标模型文件: {target_chkpt_name}")
                return
            self.load_state_dict(torch.load(target_chkpt_name, map_location=torch.device(device)))
        else:
            # 确保路径存在
            if not os.path.exists(self.chkpt_file):
                print(f"警告: 找不到模型文件: {self.chkpt_file}")
                return
            self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))