import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from tqdm import tqdm
from gym.wrappers import RecordVideo
import os
#设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
SEED = 42
set_seed(seed=SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#离散化动作空间
# [steering, gas, brake] - 原始动作空间的三个维度
# 离散化为21个动作: 不动, 左转, 右转, 加速, 刹车
DISCRETE_ACTIONS = [
    [0.0, 0.0, 0.0],   # do nothing
    [-1.0, 0.7, 0.0],  # turn left hard, accelerate hard
    [-0.5, 0.5, 0.0],  # turn left soft, accelerate soft
    [-1.0, 0.5, 0.0],  # turn left hard, accelerate soft
    [-0.5, 0.7, 0.0],  # turn left soft, accelerate hard
    [1.0, 0.5, 0.0],   # turn right hard, accelerate hard
    [0.5, 0.5, 0.0],   # turn right soft, accelerate soft
    [1.0, 0.5, 0.0],   # turn right hard, accelerate soft
    [0.5, 0.7, 0.0],   # turn right soft, accelerate hard
    [-1.0, 0.0, 1.0],  # turn left hard, brake hard
    [-0.5, 0.0, 0.5], # turn left soft, brake soft
    [-1.0, 0.0, 0.5], # turn left hard, brake soft
    [-0.5, 0.0, 1.0], # turn left soft, brake hard
    [1.0, 0.0, 1.0],   # turn right hard, brake hard
    [0.5, 0.0, 0.5],   # turn right soft, brake soft
    [1.0, 0.0, 0.5],   # turn right hard, brake soft
    [0.5, 0.0, 1.0],   # turn right soft, brake hard
    [0.0, 1.0, 0.0],   # go straight, accelerate hard
    [0.0, 0.5, 0.0],   # go straight, accelerate soft
    [0.0, 0.0, 1.0],   # go straight, brake hard
    [0.0, 0.0, 0.5]    # go straight, brake soft
]

# 图像预处理
def preprocess_observation(observation):
    # 调整图像大小
    observation = cv2.resize(observation, (84, 84))
    # 转换为灰度图
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    # 标准化
    observation = observation / 255.0
    return observation

# DQN网络定义
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出特征的尺寸
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
# DQN Agent
class DQNAgent:
    def __init__(self, state_shape, n_actions, lr=0.0002):
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        # 策略网络和目标网络
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(10000)
        
        # 探索率初始化
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 20000
        self.epsilon = self.epsilon_start
        self.steps_done = 0
        
    def select_action(self, state):
        # ε-贪婪策略选择动作
        sample = random.random()
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if sample > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.n_actions)
    
    def learn(self, batch_size, gamma=0.99):
        if len(self.buffer) < batch_size:
            return
        
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算下一状态的最大Q值（目标网络）
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # 计算期望Q值
        expected_q_values = rewards + gamma * next_q_values * (1 - dones)
        expected_q_values = expected_q_values.unsqueeze(1)
        
        # 计算Huber损失
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

# 堆叠帧以捕获时间信息
class FrameStack:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(frame)
    
    def push(self, frame):
        self.frames.append(frame)
    
    def get_state(self):
        return np.array(self.frames)
    
def train_dqn():
    #初始化环境
    env = gym.make('CarRacing-v2',continuous=True)
    env.reset(seed=SEED)
    #堆叠4帧以捕获时间信息
    NUM_FRAMES = 4
    frame_stack = FrameStack(NUM_FRAMES)
    #初始化智能体
    n_actions = len(DISCRETE_ACTIONS)
    state_shape = (NUM_FRAMES, 84, 84)  # 堆叠帧后的状态形状
    agent = DQNAgent(state_shape, n_actions)
    #训练参数
    num_episodes = 500
    batch_size = 64
    target_update_freq = 10  # 每10个episode更新一次目标网络
    episode_rewards = []
    all_rewards = []

    best_reward = -float('inf')

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        frame = preprocess_observation(obs)
        frame_stack.reset(frame)
        state = frame_stack.get_state()

        episode_reward = 0
        losses = []

        for t in range(1000): 
            action = agent.select_action(state)
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(DISCRETE_ACTIONS[action])
            frame = preprocess_observation(obs)
            frame_stack.push(frame)
            next_state = frame_stack.get_state()
            reward = min(max(reward, -1.0), 1.0)  # 奖励归一化
            episode_reward += reward
            done = terminated or truncated
            # 存储经验
            agent.buffer.push(state, action, reward, next_state, done)
            if len(agent.buffer) > batch_size:
                loss = agent.learn(batch_size)
                if loss is not None:
                    losses.append(loss)
            state = next_state
            if done:
                break
        episode_rewards.append(episode_reward)
        all_rewards.append(episode_reward)
        avg_loss = np.mean(losses) if losses else 0

        # 更新目标网络
        if episode % target_update_freq == 0:
            agent.update_target_net()
            print(f"Updated target network at episode {episode+1}")
        print(f"Episode {episode+1}: Reward: {episode_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model("best_dqn_model.pth")
            print(f"New best model saved with reward: {best_reward:.2f}")

        #每10个episode绘制一次奖励曲线
        if (episode + 1) % 50  == 0 & episode > 200:
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.title('DQN Training on CarRacing-v2')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(f'./result/reward_curve_episode_{episode+1}.png')
            plt.close()
        #每一百个episode保存一次模型
        if (episode + 1) % 100 == 0:
            agent.save_model(f"checkpoint_dqn_model_episode_{episode+1}.pth")
            print(f"Checkpoint saved at episode {episode+1}")
    # 绘制所有奖励的曲线
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards)
    plt.title('DQN Training on CarRacing-v2')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('final_reward_curve.png')
    plt.close()
    agent.save_model("final_dqn_model.pth")
    env.close()
    return all_rewards

def evaluate_model(model_path='best_dqn_model.pth', num_episodes=10, record_video=True):
    """
    评估训练好的模型并保存评估视频
    
    参数:
    - model_path: 模型权重文件路径
    - num_episodes: 评估的episode数量
    - record_video: 是否录制视频
    """
    # 创建视频保存目录
    video_dir = "evaluation_videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # 初始化环境，如果录制视频则使用RecordVideo包装器
    if record_video:
        env = gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array')
        env = RecordVideo(env, video_dir, 
                         episode_trigger=lambda episode_id: True,  # 录制所有episode
                         name_prefix="car_racing_dqn_eval")
    else:
        env = gym.make('CarRacing-v2', continuous=True, render_mode='human')
    
    
    # 堆叠4帧作为状态
    NUM_FRAMES = 4
    frame_stack = FrameStack(NUM_FRAMES)
    
    # 初始化智能体
    n_actions = len(DISCRETE_ACTIONS)
    state_shape = (NUM_FRAMES, 84, 84)
    agent = DQNAgent(state_shape, n_actions)
    
    # 加载训练好的模型
    agent.load_model(model_path)
    agent.policy_net.eval()  
    # 评估时设置为接近零的探索率，以获得确定性的策略行为
    agent.epsilon = 0.001
    
    # 评估指标
    total_rewards = []
    steps_per_episode = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        frame = preprocess_observation(obs)
        frame_stack.reset(frame)
        state = frame_stack.get_state()
        
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # 选择动作
            action_idx = agent.select_action(state)
            action = DISCRETE_ACTIONS[action_idx]
            
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            reward = min(max(reward, -1.0), 1.0)  # 奖励归一化
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            # 处理观测
            frame = preprocess_observation(obs)
            frame_stack.push(frame)
            state = frame_stack.get_state()
            
            # 限制每个episode的最大步数，防止无限运行
            if step_count >= 1000:
                break
        
        total_rewards.append(episode_reward)
        steps_per_episode.append(step_count)
        print(f"Evaluation Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    env.close()
    
    # 计算评估指标
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(steps_per_episode)
    
    print("\n--- Evaluation Results ---")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    
    # 绘制每个episode的奖励分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_episodes+1), total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Performance Evaluation')
    plt.axhline(y=avg_reward, color='r', linestyle='-', label=f'Avg: {avg_reward:.2f}')
    plt.legend()
    plt.savefig(f'{video_dir}/evaluation_rewards.png')
    plt.show()
    
    return avg_reward, total_rewards
        

if __name__ == "__main__":
    
    train_dqn()
    # 评估模型
    evaluate_model(model_path='final_dqn_model.pth', num_episodes=10, record_video=True)
    

