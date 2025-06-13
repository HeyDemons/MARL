import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import imageio
from collections import deque
import os 
import matplotlib.pyplot as plt

# --- 实验参数 ---
ENV_NAME = "CartPole-v1"
HIDDEN_DIM = 128
LEARNING_RATE = 0.01
GAMMA = 0.99
NUM_EPISODES = 1000
MAX_EPISODE_STEPS = 500
PRINT_INTERVAL = 100
GIF_FILENAME = "cartpole_reinforce.gif"
REWARD_PLOT_FILENAME = "cartpole_reinforce_rewards.png" 
PLOT_SAVE_PATH = "./" 

# 确定计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 设计策略梯度神经网络 ---
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_logits = self.fc2(x)
        return action_logits

# 初始化环境和网络
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = Policy(state_dim, action_dim, HIDDEN_DIM).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

def select_action(state_tensor):
    action_logits = policy_net(state_tensor)
    action_dist = Categorical(logits=action_logits)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return action.item(), log_prob

def train():
    print(f"Starting training for {NUM_EPISODES} episodes...")
    total_rewards_deque = deque(maxlen=100)
    all_episode_rewards = [] # <-- 新增: 记录每个回合的总奖励

    for episode in range(NUM_EPISODES):
        state, info = env.reset()
        episode_rewards_list_for_return_calc = [] # Renamed to avoid confusion
        episode_log_probs = []
        current_episode_reward_sum = 0

        for t in range(MAX_EPISODE_STEPS):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action, log_prob = select_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_rewards_list_for_return_calc.append(reward)
            episode_log_probs.append(log_prob)
            current_episode_reward_sum += reward
            state = next_state

            if terminated or truncated:
                break
        
        all_episode_rewards.append(current_episode_reward_sum) # <-- 新增: 记录当前回合总奖励
        total_rewards_deque.append(current_episode_reward_sum)
        avg_reward = np.mean(total_rewards_deque)

        returns = deque()
        discounted_sum = 0
        for r in reversed(episode_rewards_list_for_return_calc):
            discounted_sum = r + GAMMA * discounted_sum
            returns.appendleft(discounted_sum)
        
        returns = torch.tensor(list(returns), dtype=torch.float32).to(device)
        if len(returns) > 1: 
             returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        elif len(returns) == 1:
            returns = (returns - returns.mean()) / (1e-9) 

        policy_loss = []
        for log_prob, R in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        optimizer.zero_grad()
        if policy_loss: # Ensure policy_loss is not empty
            loss = torch.stack(policy_loss).sum() 
            loss.backward()
            optimizer.step()
        else:
            loss = torch.tensor(0.0) # No loss if episode ended immediately

        if (episode + 1) % PRINT_INTERVAL == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES} | Last Reward: {current_episode_reward_sum:.2f} | Avg Reward (Last 100): {avg_reward:.2f} | Loss: {loss.item():.4f}")
        
        if avg_reward >= 475.0 and len(total_rewards_deque) == 100:
            print(f"Environment solved in {episode+1} episodes! Average reward: {avg_reward:.2f}")
            # Continue training for the full NUM_EPISODES to see the full curve unless explicitly stopped
            break 
            
    print("Training finished.")
    return all_episode_rewards 

def plot_rewards(episode_rewards, filename="reward_plot.png"):
    """
    绘制并保存奖励曲线图。
    Args:
        episode_rewards (list): 包含每个回合总奖励的列表。
        filename (str): 保存图表的文件名。
    """
    print(f"Plotting rewards to {filename}...")
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Reward per Episode')
    
    # 计算并绘制移动平均奖励 (例如，每100个回合的窗口)
    if len(episode_rewards) >= 100:
        window_size = 100
        moving_avg_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg_rewards, label='Moving Average (10 episodes)', color='orange', linestyle='--')
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Training")
    plt.legend()
    plt.grid(True)
    # 确保保存路径存在
    if not os.path.exists(PLOT_SAVE_PATH):
        os.makedirs(PLOT_SAVE_PATH)
    plot_full_path = os.path.join(PLOT_SAVE_PATH, filename)
    plt.savefig(plot_full_path)
    plt.close() 
    print(f"Reward plot saved to {plot_full_path}")


def save_game_animation():
    print("Saving trained agent's gameplay as GIF...")
    if not os.path.exists(PLOT_SAVE_PATH): # GIF也保存到同一路径
        os.makedirs(PLOT_SAVE_PATH)
    gif_full_path = os.path.join(PLOT_SAVE_PATH, GIF_FILENAME)

    render_env = gym.make(ENV_NAME, render_mode="rgb_array")
    state, info = render_env.reset()
    frames = []
    total_reward = 0

    for t in range(MAX_EPISODE_STEPS):
        frame = render_env.render()
        frames.append(frame)
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_logits = policy_net(state_tensor)
            action_dist = Categorical(logits=action_logits)
            # 在评估时，有时会选择概率最大的动作，而不是采样
            # action = action_logits.argmax(dim=-1).item() 
            action = action_dist.sample().item() # 继续使用采样以保持一致性
            
        state, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    render_env.close()
    
    if frames:
        imageio.mimsave(gif_full_path, frames, duration = 20, loop=0)
        print(f"GIF saved to {gif_full_path}. Total reward in saved game: {total_reward}")
    else:
        print("No frames recorded for GIF.")


if __name__ == "__main__":
    # 确保保存路径存在
    if not os.path.exists(PLOT_SAVE_PATH):
        os.makedirs(PLOT_SAVE_PATH)
        
    episode_rewards_history = train()
    plot_rewards(episode_rewards_history, filename=REWARD_PLOT_FILENAME)
    save_game_animation()

    env.close()