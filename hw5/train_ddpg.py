import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 设置为你期望的 GPU ID

# 添加这些辅助函数
def moving_average(data, window_size=50):
    """简单移动平均"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def exponential_moving_average(data, alpha=0.1):
    """指数移动平均"""
    ema = np.zeros_like(data, dtype=float)  # 确保 ema 是浮点数以便计算
    if data.size == 0:  # 检查数组是否为空
        return ema
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

BUFFER_SIZE = int(1e6)  # 经验回放缓冲区大小
BATCH_SIZE = 128        # 小批量大小
GAMMA = 0.99            # 折扣因子
TAU = 0.005              # 用于目标参数的软更新
LR_ACTOR = 2e-4         # actor 的学习率
LR_CRITIC = 2e-3        # critic 的学习率
UPDATE_EVERY = 1        # 更新网络的频率
UPDATE_TIMES = 1        # 每次更新时更新网络的次数

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 帧预处理
def preprocess_frame(frame):
    """预处理帧：转换为灰度图，调整大小，归一化"""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0

class FrameStack:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
    
    def reset(self, frame):
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return self.get_state()
    
    def append(self, frame):
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        return np.stack(self.frames, axis=-1)
    
# Actor 网络 (策略)
class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self. max_action = max_action
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.cnn(x)
        return self.fc(x)

# Critic 网络 (价值)
class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        
        # 状态路径
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 动作路径
        self.action_fc = nn.Linear(action_dim, 512)
        
        # 组合路径
        self.fc = nn.Sequential(
            nn.Linear(3136 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, action):
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.cnn(x)
        action = self.action_fc(action)
        x = torch.cat([x, action], dim=1)
        return self.fc(x)
    

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", 
            field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.FloatTensor(np.array([e.action for e in experiences])).to(device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).unsqueeze(1).to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)

# Ornstein-Uhlenbeck 噪声
class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    

class DDPGAgent:
    def __init__(self, state_shape, action_dim, max_action):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 网络
        self.actor = Actor(action_dim, max_action).to(device)
        self.actor_target = Actor(action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic = Critic(action_dim).to(device)
        self.critic_target = Critic(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(BUFFER_SIZE)
        
        # 噪声过程
        self.noise = OUNoise(action_dim)
        
        # 训练统计
        self.total_steps = 0
        self.episode_rewards = []
        self.best_reward = -float('inf')
    
    def act(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        
        if noise:
            action += self.noise.sample()
        
        return np.clip(action, -self.max_action, self.max_action)
    
    def step(self, state, action, reward, next_state, done):
        # 在经验回放缓冲区中保存经验
        self.memory.add(state, action, reward, next_state, done)
        
        # 每 UPDATE_EVERY 个时间步学习一次
        self.total_steps += 1
        if len(self.memory) > BATCH_SIZE and self.total_steps % UPDATE_EVERY == 0:
            for _ in range(UPDATE_TIMES):
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # 更新 critic
        # 从目标模型获取预测的下一状态动作和 Q 值
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        
        # 计算当前状态的 Q 目标
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # 计算 critic 损失
        Q_expected = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # 最小化损失
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        
        # 更新 actor
        # 计算 actor 损失
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        # 最小化损失
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新目标网络
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"模型已保存至 {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"模型已从 {path} 加载")

def train(num_episodes=1000, max_t=1000):
    all_rewards = []
    # avg_rewards = [] # 在新样式中不再显式绘制，而是隐式计算或可以保留用于其他日志记录

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"ddpg_carracing_results_{timestamp}"
    model_dir = os.path.join(results_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 如果 results_dir 中不存在，则创建绘图目录
    plot_dir = results_dir 
    # if not os.path.exists(plot_dir): # results_dir 已创建
    #     os.makedirs(plot_dir)

    for i_episode in range(1, num_episodes + 1):
        frame = env.reset()[0]
        frame = preprocess_frame(frame)
        state = frame_stack.reset(frame)
        agent.noise.reset()
        episode_reward = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_frame, reward, done, truncated, _ = env.step(action)
            next_frame = preprocess_frame(next_frame)
            next_state = frame_stack.append(next_frame)
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        agent.episode_rewards.append(episode_reward)  

        all_rewards.append(episode_reward)
        # avg_reward = np.mean(all_rewards[-100:]) # 用于日志记录的原始计算
        # avg_rewards.append(avg_reward) # 用于日志记录的原始列表

        current_avg_reward = np.mean(all_rewards[-100:]) # 用于打印输出和保存最佳模型
        
        if episode_reward > agent.best_reward:
            agent.best_reward = episode_reward
            agent.save_model(os.path.join(model_dir, f"best_model.pth"))
        
        print(f"回合 {i_episode}/{num_episodes}, 奖励: {episode_reward:.2f}, 平均奖励 (最近100回合): {current_avg_reward:.2f}")

    # 保存最终模型
    agent.save_model(os.path.join(model_dir, "final_model.pth"))
    
    # 保存训练结果图
    episodes_axis = np.arange(1, len(all_rewards) + 1)
    all_rewards_np = np.array(all_rewards)

    window_size = 50  # 根据您的示例
    alpha = 0.1       # 根据您的示例

    # 计算平滑数据
    # 确保有足够的数据进行移动平均
    if len(all_rewards_np) >= window_size:
        ma_rewards = moving_average(all_rewards_np, window_size)
        ma_episodes_axis = episodes_axis[window_size-1:]
    else:
        ma_rewards = np.array([]) # 如果数据不足则为空数组
        ma_episodes_axis = np.array([])

    if len(all_rewards_np) > 0:
        ema_rewards = exponential_moving_average(all_rewards_np, alpha)
    else:
        ema_rewards = np.array([]) # 如果没有数据则为空数组

    # 1. 移动平均图
    plt.figure(figsize=(12, 6))
    plt.plot(episodes_axis, all_rewards_np, 'lightgray', alpha=0.5, label='original rewards')
    if ma_rewards.any(): # 检查 ma_rewards 是否不为空
        plt.plot(ma_episodes_axis, ma_rewards, 'b-', linewidth=2, label=f'moving average (w={window_size})')
    plt.title('episode rewards - moving average')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    save_path_ma = os.path.join(plot_dir, f'rewards_plot_ma_{timestamp}.png')
    plt.savefig(save_path_ma, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"移动平均图已保存至 {save_path_ma}")

    # 2. 指数移动平均图
    plt.figure(figsize=(12, 6))
    plt.plot(episodes_axis, all_rewards_np, 'lightgray', alpha=0.5, label='original rewards')
    if ema_rewards.any(): # 检查 ema_rewards 是否不为空
        plt.plot(episodes_axis, ema_rewards, 'r-', linewidth=2, label=f'Exponential Moving Average (alpha={alpha})')
        plt.title('Episode Rewards - Exponential Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    save_path_ema = os.path.join(plot_dir, f'rewards_plot_ema_{timestamp}.png')
    plt.savefig(save_path_ema, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"指数移动平均图已保存至 {save_path_ema}")


def record_video(env, agent, folder='ddpg_carracing_results_20250612-230832'):
    """记录 agent 的性能视频"""
    video_folder = os.path.join(folder, "videos")
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    # 创建视频录制环境
    video_env = RecordVideo(
        gym.make('CarRacing-v3', render_mode='rgb_array'),
        video_folder=video_folder,
        episode_trigger=lambda x: True,  # 记录每一集
        name_prefix="ddpg-carracing"
    )
    
    # 加载最佳模型
    model_path = os.path.join(folder, "models","final_model.pth") # Corrected path
    agent.load_model(model_path)
    agent.noise.reset()
    agent.noise.sigma = 0  # 评估时禁用噪声
    
    for episode in range(3):  
        frame = video_env.reset()[0]
        frame = preprocess_frame(frame)
        state = frame_stack.reset(frame)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, noise=False)
            next_frame, reward, done, truncated, _ = video_env.step(action)
            next_frame = preprocess_frame(next_frame)
            state = frame_stack.append(next_frame)
            total_reward += reward
            done = done or truncated
        
        print(f"测试回合: {episode}, 奖励: {total_reward:.2f}")
    
    video_env.close()

if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode='rgb_array')
    frame_stack = FrameStack(stack_size=4)
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = DDPGAgent(state_shape=(84, 84, 4), action_dim=action_dim, max_action=max_action) 

    # 训练 agent
    # train(num_episodes=500, max_t=1000)
    record_video(env, agent) # 取消注释以在训练后录制视频

