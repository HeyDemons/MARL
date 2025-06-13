import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from gym import spaces
from tqdm import tqdm

# 1. 构建自定义环境"一维导航"
class OneDNavigation(gym.Env):
    def __init__(self):
        super(OneDNavigation, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: 左移, 1: 右移
        self.observation_space = spaces.Discrete(10)  # 10个位置状态
        self.max_steps = 20  # 最大步数限制
        self.reset()
        
    def reset(self):
        self.position = 0  # 初始位置在起点
        self.steps = 0
        return self.position
    
    def step(self, action):
        self.steps += 1
        if action == 0:  # 左移
            self.position = max(0, self.position - 1)
        else:  # 右移
            self.position = min(9, self.position + 1)
        
        # 奖励设计
        if self.position == 9:  # 到达终点
            reward = 100
            done = True
        elif self.position == 5:  # 障碍物
            reward = -50
            done = False
        else:  # 其他情况
            reward = -1
            done = False
        
        # 超过最大步数也结束
        if self.steps >= self.max_steps:
            done = True
            
        return self.position, reward, done, {"position": self.position}
    
    def render(self, mode='human'):
        grid = ['_'] * 10
        grid[self.position] = 'A'  # 智能体
        grid[5] = 'X'  # 障碍物
        grid[9] = 'G'  # 目标
        print('|' + '|'.join(grid) + '|')

# 2. 实现A2C模型
class A2C(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(A2C, self).__init__()
        # 共享的特征提取层
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor网络：输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络：输出状态价值
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        features = self.feature(x)
        action_probs = self.actor(features)
        state_values = self.critic(features)
        return action_probs, state_values

# 状态one-hot编码
def one_hot_encode(state, num_states=10):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return one_hot

# 3. 实现A2C训练流程
def train(env, model, optimizer, n_episodes=1000, gamma=0.99, entropy_coef=0.01, max_grad_norm=0.5):
    episode_rewards = []
    success_count = 0
    
    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        state = one_hot_encode(state)
        log_probs = []
        values = []
        rewards = []
        entropies = []
        done = False
        episode_reward = 0
        
        while not done:
            # 收集经验
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = model(state_tensor)
            m = Categorical(action_probs)
            action = m.sample()
            next_state, reward, done, info = env.step(action.item())
            next_state = one_hot_encode(next_state)
            
            log_prob = m.log_prob(action)
            entropy = m.entropy()
            
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            entropies.append(entropy)
            
            state = next_state
            episode_reward += reward
            
            # 检查是否成功到达终点
            if info["position"] == 9:
                success_count += 1
        
        # 计算回报和优势函数
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).unsqueeze(1)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        entropies = torch.cat(entropies)
        
        # 计算优势函数 A(s,a) = R - V(s)
        advantages = returns - values.detach()
        
        # 计算Actor和Critic损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss
        
        # 更新模型参数
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            success_rate = success_count / 50 if episode >= 50 else success_count / (episode + 1)
            print(f"Episode {episode+1}, 平均奖励: {avg_reward:.2f}, 成功率: {success_rate:.2f}")
            success_count = 0  # 重置计数
    
    return episode_rewards

# 评估训练好的策略
def evaluate(env, model, n_episodes=10):
    total_rewards = []
    success_count = 0
    step_counts = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode+1}:")
        env.render()
        
        while not done:
            state_tensor = torch.FloatTensor(one_hot_encode(state)).unsqueeze(0)
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            state = next_state
            env.render()
            
            if info["position"] == 9:  # 成功到达终点
                success_count += 1
        
        step_counts.append(steps)
        total_rewards.append(total_reward)
        print(f"步数: {steps}, 总奖励: {total_reward}")
    
    success_rate = success_count / n_episodes
    avg_steps = np.mean(step_counts)
    avg_reward = np.mean(total_rewards)
    
    print(f"\n评估结果:")
    print(f"成功率: {success_rate:.2f}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"平均奖励: {avg_reward:.2f}")
    
    return avg_reward, success_rate, avg_steps

# 可视化策略
def visualize_policy(model):
    plt.figure(figsize=(12, 4))
    
    # 获取每个位置的最优动作和状态价值
    positions = np.arange(10)
    best_actions = []
    state_values = []
    
    for pos in positions:
        state = one_hot_encode(pos)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = model(state_tensor)
        best_action = torch.argmax(action_probs).item()
        best_actions.append("←" if best_action == 0 else "→")
        state_values.append(value.item())
    
    # 绘制最优策略
    plt.subplot(1, 2, 1)
    plt.bar(positions, [1] * 10, color=['lightblue'] * 10)
    plt.bar(5, 1, color='red', alpha=0.7)  # 障碍物
    plt.bar(9, 1, color='green', alpha=0.7)  # 目标
    
    for i, action in enumerate(best_actions):
        plt.text(i, 0.5, action, ha='center', va='center', fontsize=15)
    
    plt.title('Optimal Policy')
    plt.xlabel('Position')
    plt.xticks(positions)
    plt.yticks([])
    
    # 绘制状态价值
    plt.subplot(1, 2, 2)
    plt.bar(positions, state_values, color='lightgreen')
    plt.title('State Value')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.xticks(positions)
    
    plt.tight_layout()
    plt.savefig('a2c_policy_visualization.png')
    plt.show()


def main():
    # 设置随机种子以确保实验可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    env = OneDNavigation()
    input_size = 10  # One-hot编码状态维度
    hidden_size = 64  # 隐藏层大小
    output_size = 2   # 两个动作（左移或右移）
    
    model = A2C(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    print("开始训练A2C模型...")
    episode_rewards = train(env, model, optimizer, n_episodes=500)
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Training Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('A2C Training Reward Curve')
    plt.savefig('a2c_reward_curve.png')
    plt.show()
    
    # 评估
    print("\n评估训练好的策略:")
    avg_reward, success_rate, avg_steps = evaluate(env, model)
    
    # 可视化最优策略
    visualize_policy(model)
    
    # 保存模型
    torch.save(model.state_dict(), 'a2c_model.pth')

if __name__ == "__main__":
    main()