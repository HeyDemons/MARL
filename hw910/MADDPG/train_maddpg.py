import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pettingzoo.mpe import simple_adversary_v3
import wandb
import os
from tqdm import tqdm
from multiprocessing import Process, Pipe
from datetime import datetime
import imageio

class EnvWrapper:
    """
    一个用于PettingZoo MPE环境的包装器，以标准化接口。
    此类处理观察值处理和动作空间转换。
    """
    def __init__(self, render_mode="human"):
        """
        初始化环境包装器。

        Args:
            render_mode (str): 环境的渲染模式。
                               对于实时可视化使用'human'，对于视频录制使用'rgb_array'。
        """
        # 为对抗者的观察值添加填充，以匹配其他智能体的观察维度。
        self.padding = np.zeros(2)
        self.env = simple_adversary_v3.parallel_env(render_mode=render_mode, continuous_actions=True)

    def process_obs(self, obs):
        """填充对抗者的观察值，使其形状与优秀智能体相同。"""
        obs['adversary_0'] = np.concatenate([self.padding, obs['adversary_0']], axis=0)
        return obs

    def reset(self, seed=None):
        """重置环境并以列表形式返回观察值和信息。"""
        obs, infos = self.env.reset(seed=seed)
        obs = self.process_obs(obs)
        return self.dict_to_list(obs), self.dict_to_list(infos)

    def step(self, action):
        """
        在环境中执行一步。
        输入动作范围为[-1, 1]，并被转换为MPE环境期望的5维离散动作空间。
        
        Args:
            action (np.ndarray): 一个形状为(num_agents, 2)的动作数组。
        
        Returns:
            Tuple: 包含下一个观察、奖励、完成标志、截断标志和信息的元组。
        """
        # 策略网络输出在[-1, 1]范围内的动作。这需要
        # 转换为环境的5维动作空间。
        # [no_op, move_left, move_right, move_down, move_up]
        full_action = np.zeros((3, 5), dtype=np.float32)
        full_action[:, 1] = -action[:, 0]  #向左移动
        full_action[:, 2] = action[:, 0]   #向右移动
        full_action[:, 3] = -action[:, 1]  #向下移动
        full_action[:, 4] = action[:, 1]   #向上移动
        
        # 动作是力，所以它们必须是非负的。
        full_action = np.where(full_action < 0, 0, full_action)
        
        obs, rewards, dones, tructs, infos = self.env.step(self.list_to_dict(full_action))
        obs = self.process_obs(obs)
        return self.dict_to_list(obs), self.dict_to_list(rewards), self.dict_to_list(dones), self.dict_to_list(tructs), self.dict_to_list(infos)

    def list_to_dict(self, data_list):
        """将智能体数据列表转换为字典。"""
        return {
            'adversary_0': data_list[0],
            'agent_0': data_list[1],
            'agent_1': data_list[2]
        }

    def dict_to_list(self, data_dict):
        """将智能体数据字典转换为列表。"""
        return [data_dict['adversary_0'], data_dict['agent_0'], data_dict['agent_1']]
    
    # --- FIX START: Add the render method ---
    def render(self):
        """Renders the environment by calling the underlying environment's render method."""
        return self.env.render()
    # --- FIX END ---

    def close(self):
        """关闭环境。"""
        self.env.close()

def worker(remote, parent_remote, env_fn):
    """
    多处理工作者的目标函数。它在一个单独的进程中运行一个环境。
    """
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, truct, info = env.step(data)
            remote.send((obs, reward, done, truct, info))
        elif cmd == 'reset':
            obs, _ = env.reset()
            remote.send(obs)
        elif cmd == 'close':
            env.close()
            remote.close()
            break


class ParallelEnv:
    """
    一个使用多处理并行运行多个环境实例的类。
    这个“向量化”的环境允许更快地收集经验。
    """
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        
        env_fn = lambda: EnvWrapper(render_mode='rgb_array') # 并行处理时使用rgb_array

        self.ps = [Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        对所有并行环境执行一步。

        Args:
            actions (np.ndarray): 形状为 (n_envs, n_agents, action_dim) 的动作数组。
        
        Returns:
            Tuple: 包含堆叠的观察、奖励、完成标志、截断标志和信息的元组。
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, tructs, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), np.array(tructs), infos

    def reset(self):
        """重置所有并行环境。"""
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.array([remote.recv() for remote in self.remotes])

    def close(self):
        """关闭所有并行环境。"""
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

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
    def __init__(self,state_dim,hidden_dim,action_dim,agent_id,num_agent,num_env,actor_lr,critic_lr,gamma,explore_rate,explore_rate_decay,min_explore_rate,update_gap,device):
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNet((state_dim + action_dim) * num_agent, hidden_dim).to(device)
        self.target_actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = ValueNet((state_dim + action_dim) * num_agent, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.gamma = gamma
        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_rate_decay
        self.min_explore_rate = min_explore_rate
        self.update_gap = update_gap
        self.agent_id = agent_id
        self.num_agent = num_agent
        self.num_env = num_env
        self.device = device
        self.action_dim = action_dim
        self.update_step = 0
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
        self.explore_rate = max(self.min_explore_rate, self.explore_rate * self.explore_rate_decay)

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

    def __init__(self, state_dim, hidden_dim, action_dim, num_agents, num_env, actor_lr, critic_lr, gamma, explore_cfg, update_gap, device):
        self.agents = [
            DDPG(state_dim, hidden_dim, action_dim, agent_id, num_agents, num_env, actor_lr, critic_lr, gamma,
                 explore_cfg['initial'], explore_cfg['decay'], explore_cfg['min'],
                 update_gap, device)
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
        """设置日志记录和WandB运行目录。"""
        if wandb.run is None:
            wandb.init(
                project="MADDPG",
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # 使用日期时间作为运行名称
                config={
                    "state_dim": self.agents[0].actor.layer1[0].in_features,
                    "action_dim": self.agents[0].actor.fc.out_features,
                    "num_agents": self.num_agents
                }
            )
        self.run_dir = os.path.join("runs", "maddpg", wandb.run.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = wandb

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
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)  # 优化张量创建
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
        for i, agent in enumerate(self.agents):
            actor_loss, critic_loss = agent.update(
                states_tensor[:, i, :], shared_states, shared_actions, online_actions,
                rewards_tensor[:, i],  # 修正索引，移除多余的维度
                shared_next_states, online_next_actions_cat,
                dones_tensor[:, i]
            )
            # Log losses
            self.writer.log({
                f"agent-{i}/actor_loss": actor_loss,
                f"agent-{i}/critic_loss": critic_loss,
                f"agent-{i}/explore_rate": agent.explore_rate,
                "episode": i_episode
            })



def run_training_loop(env, agent, replay_buffer, config):
    '''主训练循环。'''
    best_avg_reward = -np.inf

    with tqdm(total=config['num_episodes'], desc='训练进度') as pbar:
        for i_episode in range(config['num_episodes']):
            episode_returns = np.zeros((config['num_env'], config['num_agents']))
            state = env.reset() # (num_env, num_agents, state_dim)
            
            # 在一个回合（episode）内运行
            while True:
                action = agent.take_action(state, explore=True)
                next_state, reward, done, truct, _ = env.step(action)
                
                replay_buffer.add(state, action, reward, next_state, truct)
                state = next_state
                episode_returns += reward 
                
                # 检查是否有任何环境完成
                if np.any(done) or np.any(truct):
                    break
            
            # 在收集到足够的经验后开始训练
            if len(replay_buffer) >= replay_buffer.min_size:
                for _ in range(config['update_iterations']):
                    states, actions, rewards, next_states, dones = replay_buffer.sample()
                    agent.update(states, actions, rewards, next_states, dones, i_episode)
            
            # 记录奖励
            episode_rewards = np.mean(episode_returns, axis=0)
            adversary_return = episode_rewards[0] 
            good_agents_return = np.mean(episode_rewards[1:])
            if(good_agents_return > best_avg_reward):  # 修正变量名
                best_avg_reward = good_agents_return
                agent.save_models(f"_best")
            
            wandb.log({
                "episode": i_episode,
                "adversary_return": adversary_return,
                "good_agents_return": good_agents_return,
            })
            pbar.set_postfix({
                'episode': i_episode,
                'adversary_return': adversary_return,
                'good_agents_return': good_agents_return,
                'best_avg_return': best_avg_reward
            })
            pbar.update(1)  # 每次循环更新进度条
    agent.save_models("_last")
    print(f"训练完成，最佳平均奖励: {best_avg_reward:.2f}")


def evaluate_and_record(agent, config, model_load_path):
    """
    评估训练好的智能体并录制其性能视频。
    """
    print("开始评估并录制视频...")
    agent.load_models(model_load_path, "_best")  # 加载最佳模型
    # 设置用于录制的评估环境
    eval_env = EnvWrapper(render_mode="rgb_array")
    state, _ = eval_env.reset(seed=1)  # 重置环境并设置随机种子
    # 包装环境以录制视频
    video_dir = os.path.join(model_load_path, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # --- 添加下面这行代码 ---
    eval_env.env.render_mode = "rgb_array"
    frames =[]
    done = False
    for i in range(25):
        state_for_agent = np.expand_dims(state, axis=0)  # 添加批次维度
        frames.append(eval_env.render())  # 记录当前帧
        action = agent.take_action(state_for_agent, explore=False)
        next_state, reward, done, truct, _ = eval_env.step(action[0])  # 取第一个环境的动作
        state = next_state
        # if done or truct:
        #     print(f"评估回合 {i+1} 完成。")
        #     break
    eval_env.close()
    # save gif
    gif_path = 'evaluation.gif'
    imageio.mimsave(gif_path, frames, duration=33)
    print(f"评估视频已保存到 {gif_path}")

    


if __name__ == "__main__":
    # --- 配置 ---
    config = {
        # 环境和智能体设置
        'state_dim': 10,
        'action_dim': 2,
        'hidden_dim': 128,
        'num_agents': 3,
        'num_env': 20, # 并行环境的数量
        
        # 训练超参数
        'actor_lr': 5e-4,
        'critic_lr': 5e-4,
        'gamma': 0.99,
        'update_gap': 100, # 目标网络更新之间的步数
        'num_episodes': 100000,
        'update_iterations': 1, # 每个回合的更新次数

        # 探索
        'explore_cfg': {
            'initial': 1.0,
            'decay': 0.99995,
            'min': 0.001,
        },
        
        # 回放缓冲区
        'replay_buffer': {
            'min_size': 2500,
            'max_size': 100000,
            'batch_size': 1024,
        },

        # 评估
        'eval_episodes': 5, # 用于评估视频的回合数
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")


    # --- 可复现性 ---
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    agent_args = {
        'state_dim': config['state_dim'],
        'hidden_dim': config['hidden_dim'],
        'action_dim': config['action_dim'],
        'num_agents': config['num_agents'],
        'actor_lr': config['actor_lr'],
        'critic_lr': config['critic_lr'],
        'gamma': config['gamma'],
        'explore_cfg': config['explore_cfg'],
        'update_gap': config['update_gap'],
        'device': device
    }

    # --- 训练 ---
    # replay = ReplayBuffer(**config['replay_buffer'])
    # env = ParallelEnv(n_envs=config['num_env'])
    # agent = MADDPG(num_env=config['num_env'], **agent_args)
    
    # run_training_loop(env, agent, replay, config)
    # env.close()

    # --- 训练后自动评估 ---
    print("\n--- 训练完成，开始自动评估最佳模型 ---")
    eval_agent = MADDPG(num_env=1, **agent_args)
    evaluate_and_record(eval_agent, config, model_load_path='./runs/maddpg/run_20250609_160954')








