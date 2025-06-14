import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer import ReplayBuffer
from qmix_mpe import QMIX_MPE
from pettingzoo.mpe import simple_spread_v3

def make_env(episode_limit, render_mode="None"):
    env = simple_spread_v3.parallel_env(N=3, max_cycles=episode_limit,
                                        local_ratio=0.5, render_mode=render_mode, continuous_actions=False)
    env.reset(seed=42)
    return env

class Runner_QMIX_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = make_env(self.args.episode_limit, render_mode=args.render_mode)
        self.args.N = self.env.max_num_agents
        self.args.obs_dim_n = [self.env.observation_spaces[agent].shape[0] for agent in self.env.agents]
        self.args.action_dim_n = [self.env.action_spaces[agent].n for agent in self.env.agents]
        
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = sum(self.args.obs_dim_n)

        self.agent_n = QMIX_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.writer = SummaryWriter(log_dir=f'runs/QMIX/QMIX_env_{self.env_name}_number_{self.number}_seed_{self.seed}')
        self.evaluate_rewards = []
        self.total_steps = 0

    def run(self):
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps % self.args.evaluate_freq == 0:
                self.evaluate_policy()

            episode_reward, episode_steps = self.run_episode_mpe()
            self.total_steps += episode_steps
            
            # Train every N episodes
            if self.replay_buffer.counter > 0 and self.replay_buffer.counter % self.args.train_freq_episode == 0:
                for _ in range(self.args.train_epochs):
                     self.agent_n.train(self.replay_buffer, self.total_steps)

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print(f"total_steps: {self.total_steps} \t evaluate_reward: {evaluate_reward}")
        self.writer.add_scalar(f'evaluate_step_rewards_{self.env_name}', evaluate_reward, global_step=self.total_steps)
        np.save(f'./data_train/QMIX_env_{self.env_name}_number_{self.number}_seed_{self.seed}.npy', np.array(self.evaluate_rewards))

    def run_episode_mpe(self, evaluate=False):
            episode_reward = 0
            episode_buffer = {
                'obs_n': np.zeros([self.args.episode_limit, self.args.N, self.args.obs_dim]),
                's': np.zeros([self.args.episode_limit, self.args.state_dim]),
                'a_n': np.zeros([self.args.episode_limit, self.args.N, 1]),
                'r_n': np.zeros([self.args.episode_limit, self.args.N]),
                'obs_next_n': np.zeros([self.args.episode_limit, self.args.N, self.args.obs_dim]),
                's_next': np.zeros([self.args.episode_limit, self.args.state_dim]),
                'done_n': np.zeros([self.args.episode_limit, self.args.N])
            }
            
            observations, _ = self.env.reset()
            
            # --- 新增代码: 在回合开始时获取固定的智能体列表 ---
            agent_list = list(self.env.agents)
            # --- 新增代码结束 ---
            
            obs_n = np.array([observations[agent] for agent in agent_list]) # 使用 agent_list
            self.agent_n.eval_hidden = None 

            for t in range(self.args.episode_limit):
                s = obs_n.flatten()
                a_n = self.agent_n.choose_action(obs_n, evaluate)
                
                actions = {agent: a_n[i] for i, agent in enumerate(agent_list)} # 使用 agent_list
                obs_next, r, done, _, _ = self.env.step(actions)
                
                # --- 修改开始: 使用固定的 agent_list 来构建数组 ---
                obs_next_n = np.array([obs_next[agent] for agent in agent_list])
                s_next = obs_next_n.flatten()
                r_n = np.array([r[agent] for agent in agent_list])
                done_n = np.array([done[agent] for agent in agent_list])
                # --- 修改结束 ---

                if r: 
                    episode_reward += sum(r.values()) / len(r)

                if not evaluate:
                    episode_buffer['obs_n'][t] = obs_n
                    episode_buffer['s'][t] = s
                    episode_buffer['a_n'][t] = np.expand_dims(a_n, axis=1)
                    episode_buffer['r_n'][t] = r_n
                    episode_buffer['s_next'][t] = s_next
                    episode_buffer['obs_next_n'][t] = obs_next_n
                    episode_buffer['done_n'][t] = done_n
                
                obs_n = obs_next_n
                if all(done_n):
                    break
            
            if not evaluate:
                self.replay_buffer.store_episode(episode_buffer)

            return episode_reward, t + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for QMIX in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=int, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Evaluate times")
    
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_freq_episode", type=int, default=5, help="Train the model every N episodes")
    parser.add_argument("--train_epochs", type=int, default=1, help="Number of epochs per training interval")

    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--qmix_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the qmixer")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--grad_clip_norm", type=float, default=10.0, help="Gradient clip norm")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Frequency of updating the target network")

    parser.add_argument('--render_mode', type=str, default='None', help="Render mode")
    
    args = parser.parse_args()
    runner = Runner_QMIX_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    runner.run()