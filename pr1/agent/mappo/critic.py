import torch
import torch.nn as nn
import os
from datetime import datetime

class CriticPPO(nn.Module):
    def __init__(self, chkpt_name, chkpt_dir, in_dim, hidden_dim=128, non_linear=nn.ReLU()):
        super(CriticPPO, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.chkpt_name = chkpt_name

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, 1), # Outputs a single value
        ).apply(self.init)

    @staticmethod
    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            m.bias.data.fill_(0.0)
    
    def forward(self, x):
        return self.net(x)
    
    def save_checkpoint(self, timestamp=False):
        save_dir = self.chkpt_dir
        if timestamp:
            current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            save_dir = os.path.join(self.chkpt_dir, current_timestamp)
        
        os.makedirs(save_dir, exist_ok=True)
        self.chkpt_file = os.path.join(save_dir, self.chkpt_name)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, device='cpu', timestamp=None):
        load_dir = self.chkpt_dir
        if timestamp and isinstance(timestamp, str):
            load_dir = os.path.join(self.chkpt_dir, timestamp)

        self.chkpt_file = os.path.join(load_dir, self.chkpt_name)
        if not os.path.exists(self.chkpt_file):
            print(f"Warning: No model file found at {self.chkpt_file}")
            return
        self.load_state_dict(torch.load(self.chkpt_file, map_location=torch.device(device)))