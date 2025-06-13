import torch
import torch.nn as nn
import torch.functional as F
import os
from datetime import datetime

class Critic(nn.Module):
    def __init__(self, chkpt_name,  chkpt_dir, in_dim, out_dim, hidden_dim = 128, non_linear = nn.ReLU()):
        super(Critic, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_name = chkpt_name

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        '''init patameters of the module'''
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)

    def save_checkpoint(self, is_target = False, timestamp = False):
        if timestamp is True:
            current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            save_dir = os.path.join(self.chkpt_dir, current_timestamp)
        else:
            save_dir = self.chkpt_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.chkpt_file = os.path.join(save_dir, self.chkpt_name)

        if is_target:
            target_chkpt_name = self.chkpt_file.replace('.pth', '_target.pth')
            torch.save(self.state_dict(), target_chkpt_name)
        else:
            torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device = 'cpu', is_target = False, timestamp = None):
        if timestamp and isinstance(timestamp, str):
            load_dir = os.path.join(self.chkpt_dir, timestamp)
        else:
            load_dir = self.chkpt_dir
        
        self.chkpt_file = os.path.join(load_dir, self.chkpt_name)

        if is_target:
            target_chkpt_name = self.chkpt_file.replace('.pth', '_target.pth')
            if not os.path.exists(target_chkpt_name):
                print(f"Warning: Could not find target model file: {target_chkpt_name}")
                return
            self.load_state_dict(torch.load(target_chkpt_name, map_location=torch.device(device)))
        else:
            if not os.path.exists(self.chkpt_file):
                print(f"Warning: Could not find model file: {self.chkpt_file}")
                return
            self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))