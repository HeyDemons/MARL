import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pettingzoo.mpe import simple_adversary_v3
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from envs import ParallelEnv, AdversaryEnvWrapper
from datetime import datetime
import imageio



# Replay buffer for storing transitions
class ReplayBuffer:
    """Replay buffer for off-policy learning."""
    def __init__(self, min_size, max_size, batch_size):
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = []
        self.batch_size = batch_size

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(len(rewards)):
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)
            self.buffer.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class PolicyNet(nn.Module):
    """Policy network for the MADDPG agent."""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return F.tanh(self.fc(x))  # Tanh for continuous action space

class ValueNet(nn.Module):
    """Value network for the MADDPG agent."""
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc(x)  # No activation for value output 
    
class DDPG:
    def __init__(self,state_dim,hidden_dim,action_dim,agent_id,num_agent,num_env,actor_lr,critic_lr,gamma,explore_rate,min_explore_rate,update_gap,device,total_episodes):
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNet((state_dim + action_dim) * num_agent, hidden_dim).to(device)
        self.target_actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = ValueNet((state_dim + action_dim) * num_agent, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.gamma = gamma
        self.initial_explore_rate = explore_rate
        self.explore_rate = explore_rate
        self.min_explore_rate = min_explore_rate
        self.update_gap = update_gap
        self.agent_id = agent_id
        self.num_agent = num_agent
        self.num_env = num_env
        self.device = device
        self.action_dim = action_dim
        self.update_step = 0
        # 线性衰减相关参数
        self.total_episodes = total_episodes
        self.decay_episodes = int(0.9 * total_episodes)  # 90%的episodes数进行衰减
        self.current_episode = 0

    def update_explore_rate(self, episode):
        """更新探索率，使用线性衰减"""
        self.current_episode = episode
        if episode < self.decay_episodes:
            # 线性衰减：从initial_explore_rate线性衰减到min_explore_rate
            self.explore_rate = self.initial_explore_rate - (self.initial_explore_rate - self.min_explore_rate) * (episode / self.decay_episodes)
        else:
            # 超过90%的episodes后，保持最小探索率
            self.explore_rate = self.min_explore_rate

    def take_action(self, state, explore=True):
        """Select action with exploration noise."""
        if explore and random.random() < self.explore_rate:
            return np.random.uniform(-1, 1, size=(self.num_env, self.action_dim))
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        return action

    def update(self, states,shared_states,actions,online_actions,rewards,shared_next_states,online_next_actions,dones):
        critic_input = torch.cat([shared_states, actions], dim=-1) 
        q_values = self.critic(critic_input)
        #使用目标网络计算目标Q值
        with torch.no_grad():
            next_critic_input = torch.cat([shared_next_states, online_next_actions], dim=-1)
            target_q_values = self.target_critic(next_critic_input)
            # print(f"rewards shape: {rewards.shape}, dones shape: {dones.shape}")
            # 确保 rewards 和 dones 是一维张量
            rewards_reshaped = rewards.unsqueeze(1)
            dones_reshaped = dones.unsqueeze(1)
            target_q_values = rewards_reshaped + (1 - dones_reshaped) * self.gamma * target_q_values
                # Critic loss
        critic_loss = F.mse_loss(q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        # Actor使用 Critic 的梯度进行更新
        # 我们用其当前策略的输出来替换此智能体的动作
        online_actioons_copy = list(online_actions)
        online_actioons_copy[self.agent_id] = self.actor(states)
        online_actioons_copy = torch.cat(online_actioons_copy, dim=-1)
        actor_input = torch.cat([shared_states, online_actioons_copy], dim=-1)
        actor_loss = -self.critic(actor_input).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #梯度裁剪
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()
        # Update target networks
        self.update_step += 1
        if self.update_step % self.update_gap == 0:
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)


        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, target_net, source_net, tau=0.005):
        """Soft update target network parameters."""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save(self, directory, prefix):
        """保存 actor 和 critic 模型。"""
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{prefix}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{prefix}_critic.pth"))
    def load(self, directory, prefix):
        """加载 actor 和 critic 模型。"""
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{prefix}_actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{prefix}_critic.pth"), map_location=self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

class MADDPG:

    def __init__(self, state_dim, hidden_dim, action_dim, num_agents, num_env, actor_lr, critic_lr, gamma, explore_cfg, update_gap, device, total_episodes):
        self.agents = [
            DDPG(state_dim, hidden_dim, action_dim, agent_id, num_agents, num_env, actor_lr, critic_lr, gamma,
                 explore_cfg['initial'], explore_cfg['min'],
                 update_gap, device, total_episodes)
            for agent_id in range(num_agents)
        ]
        self.num_agents = num_agents
        self.device = device
        self.run_dir = ""
        self.writer = None
        self._setup_logging()

    
    def save_models(self, suffix):
        """为所有智能体保存模型。"""
        model_dir = os.path.join(self.run_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        for j, agent in enumerate(self.agents):
            agent.save(model_dir, f"agent-{j}{suffix}")

    def load_models(self, path, suffix):
        """从特定目录为所有智能体加载模型。"""
        model_dir = os.path.join(path, "models")
        print(f"从以下位置加载模型: {model_dir}")
        for j, agent in enumerate(self.agents):
            agent.load(model_dir, f"agent-{j}{suffix}")

    def _setup_logging(self):
        """设置日志记录和TensorBoard运行目录。"""
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join("runs", "maddpg", run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 创建TensorBoard writer
        log_dir = os.path.join(self.run_dir, "logs")
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard日志保存到: {log_dir}")
        print(f"运行以下命令查看TensorBoard: tensorboard --logdir={log_dir}")

    def take_action(self, states, explore=True):
        """为所有智能体选择动作。"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions = []
        for i in range(self.num_agents):
            agent_states = states_tensor[:, i, :]
            action = self.agents[i].take_action(agent_states, explore)
            actions.append(action)
        return np.array(actions).transpose(1, 0, 2)
    
    def update(self, states, actions, rewards, next_states, dones, i_episode):
        """更新所有智能体的策略和价值网络。"""
        bs = len(rewards)
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        # Prepare shared states and actions for critic input
        shared_states = states_tensor.view(bs, -1)
        shared_actions = actions_tensor.view(bs, -1)
        shared_next_states = next_states_tensor.view(bs, -1)

        online_actions = [agent.actor(states_tensor[:, i, :]).detach() for i, agent in enumerate(self.agents)]
        online_next_actions = [agent.target_actor(next_states_tensor[:, i, :]).detach() for i, agent in enumerate(self.agents)]
        online_next_actions_cat = torch.cat(online_next_actions, dim=-1)
        
        # 更新所有智能体的探索率
        for agent in self.agents:
            agent.update_explore_rate(i_episode)
        
        for i, agent in enumerate(self.agents):
            actor_loss, critic_loss = agent.update(
                states_tensor[:, i, :], shared_states, shared_actions, online_actions,
                rewards_tensor[:, i],
                shared_next_states, online_next_actions_cat,
                dones_tensor[:, i]
            )
            # 使用TensorBoard记录损失
            self.writer.add_scalar(f"agent-{i}/actor_loss", actor_loss, i_episode)
            self.writer.add_scalar(f"agent-{i}/critic_loss", critic_loss, i_episode)
            self.writer.add_scalar(f"agent-{i}/explore_rate", agent.explore_rate, i_episode)

    def close_writer(self):
        """关闭TensorBoard writer。"""
        if self.writer:
            self.writer.close()

def run_training_loop(env, agent, replay_buffer, config):
    '''主训练循环。'''
    best_avg_reward = -np.inf

    with tqdm(total=config['num_episodes'], desc='训练进度') as pbar:
        for i_episode in range(config['num_episodes']):
            episode_returns = np.zeros((config['num_env'], config['num_agents']))
            state = env.reset() # (num_env, num_agents, state_dim)
            
            for _ in range(config['episode_length']):
                action = agent.take_action(state, explore=True)
                next_state, reward, done, truct, _ = env.step(action)
                
                replay_buffer.add(state, action, reward, next_state, truct)
                state = next_state
                episode_returns += reward 
                
                if np.any(done) or np.any(truct):
                    state = env.reset()

            # 在收集到足够的经验后开始训练
            if len(replay_buffer) >= replay_buffer.min_size:
                for _ in range(config['update_iterations']):
                    states, actions, rewards, next_states, dones = replay_buffer.sample()
                    agent.update(states, actions, rewards, next_states, dones, i_episode)
            
            # 记录奖励
            episode_rewards = np.mean(episode_returns, axis=0)
            adversary_return = episode_rewards[0] 
            good_agents_return = np.mean(episode_rewards[1:])
            if(good_agents_return > best_avg_reward):
                best_avg_reward = good_agents_return
                agent.save_models(f"_best")
            
            # 使用TensorBoard记录指标
            agent.writer.add_scalar("episode/adversary_return", adversary_return, i_episode)
            agent.writer.add_scalar("episode/good_agents_return", good_agents_return, i_episode)
            agent.writer.add_scalar("episode/best_avg_return", best_avg_reward, i_episode)
            
            pbar.set_postfix({
                'epo': i_episode,
                'adv_r': adversary_return,
                'good_r': good_agents_return,
                'avg_r': best_avg_reward
            })
            pbar.update(1)
    
    agent.save_models("_last")
    agent.close_writer()  # 关闭TensorBoard writer
    print(f"训练完成，最佳平均奖励: {best_avg_reward:.2f}")


def evaluate_and_record(agent, config, model_load_path):
    """
    评估训练好的智能体并录制其性能视频。
    """
    print("开始评估并录制视频...")
    agent.load_models(model_load_path, "_best")
    
    # >> 修改点: 使用新的 AdversaryEnvWrapper 创建评估环境
    eval_env = AdversaryEnvWrapper(render_mode="rgb_array")
    state, _ = eval_env.reset(seed=42)
    # << 修改点结束
    
    video_dir = os.path.join(model_load_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    gif_path = os.path.join(video_dir, 'evaluation.gif')

    frames =[]
    for _ in range(config['episode_length']): # 运行固定的步数
        frames.append(eval_env.render())
        # 添加批次维度以匹配智能体 take_action 的期望输入
        state_for_agent = np.expand_dims(state, axis=0)
        action = agent.take_action(state_for_agent, explore=False)
        # 移除批次维度以输入到 step 函数
        next_state, _, _, _, _ = eval_env.step(action[0])
        state = next_state

    eval_env.close()
    imageio.mimsave(gif_path, frames, duration=40) # duration in ms
    print(f"评估视频已保存到 {gif_path}")

    


if __name__ == "__main__":
    # --- 配置 ---
    config = {
        # >> 修改点: state_dim 将被动态确定，这里的值仅为占位符
        'state_dim': 10,
        # << 修改点结束
        'action_dim': 2,
        'hidden_dim': 128,
        'num_agents': 3,
        'num_env': 20,
        'episode_length': 25, # MPE 环境的默认回合长度
        
        'actor_lr': 5e-4,
        'critic_lr': 5e-4,
        'gamma': 0.99,
        'update_gap': 100,
        'num_episodes': 500000,
        'update_iterations': 1,

        'explore_cfg': { 'initial': 1.0, 'min': 0.001 },
        'replay_buffer': { 'min_size': 2500, 'max_size': 100000, 'batch_size': 1024 },
        'eval_episodes': 5,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 可复现性 ---
    seed = 42
    torch.manual_seed(seed)
    # ... (其余种子设置代码)

    # >> 修改点: 动态确定 state_dim 并使用新的方式初始化环境
    # 1. 创建一个临时环境来动态获取 state_dim
    print("正在确定环境维度...")
    temp_env = AdversaryEnvWrapper()
    obs, _ = temp_env.reset()
    config['state_dim'] = np.array(obs[0]).shape[0]
    temp_env.close()
    print(f"动态确定的 State Dimension: {config['state_dim']}")

    # 2. 定义环境创建函数
    env_fn = lambda: AdversaryEnvWrapper(render_mode='rgb_array')

    # 3. 初始化并行环境、回放缓冲区和智能体
    replay = ReplayBuffer(**config['replay_buffer'])
    env = ParallelEnv(env_fn=env_fn, n_envs=config['num_env'])
    
    agent_args = {
        'state_dim': config['state_dim'],
        'hidden_dim': config['hidden_dim'],
        'action_dim': config['action_dim'],
        'num_agents': config['num_agents'],
        'num_env': config['num_env'],
        'actor_lr': config['actor_lr'],
        'critic_lr': config['critic_lr'],
        'gamma': config['gamma'],
        'explore_cfg': config['explore_cfg'],
        'update_gap': config['update_gap'],
        'device': device,
        'total_episodes': config['num_episodes']  # 添加总episode数
    }
    agent = MADDPG(**agent_args)
    # << 修改点结束

    # --- 训练 ---
    run_training_loop(env, agent, replay, config)
    env.close()

    # --- 训练后自动评估 ---
    print("\n--- 训练完成，开始自动评估最佳模型 ---")
    eval_agent_args = agent_args.copy()
    eval_agent_args['num_env'] = 1 # 评估时 num_env 为 1
    eval_agent = MADDPG(**eval_agent_args)
    
    # 确保 model_load_path 指向正确的日志目录
    model_load_path = agent.run_dir 
    evaluate_and_record(eval_agent, config, model_load_path=model_load_path)