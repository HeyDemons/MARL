import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer_qmix import ReplayBuffer_QMIX
from qmix import QMIX_MPE
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

        # <--- 新增：定义 device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前使用的设备: {self.device}")

        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = make_env(self.args.episode_limit, render_mode=args.render_mode)
        self.args.N = self.env.max_num_agents
        self.args.obs_dim_n = [self.env.observation_spaces[agent].shape[0] for agent in self.env.agents]
        self.args.action_dim_n = [self.env.action_spaces[agent].n for agent in self.env.agents]
        
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        # <--- 修改：将 device 传递给 Agent
        self.agent_n = QMIX_MPE(self.args, self.device)
        self.replay_buffer = ReplayBuffer_QMIX(self.args)

        self.writer = SummaryWriter(log_dir=f'runs/QMIX/QMIX_env_{self.env_name}_number_{self.number}_seed_{self.seed}')

        self.evaluate_rewards = []
        self.total_steps = 0
        self.epsilon = args.epsilon_start

    def run(self):
        # --- 此函数主体逻辑无需修改 ---
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps % self.args.evaluate_freq == 0:
                self.evaluate_policy()

            episode_data, episode_steps, _ = self.run_episode_mpe()
            self.total_steps += episode_steps
            
            # 修改：传递 episode_steps 给 store_episode
            self.replay_buffer.store_episode(episode_data, episode_steps) 
            
            if self.replay_buffer.current_size >= self.args.batch_size:
                batch = self.replay_buffer.sample()
                self.agent_n.train(batch)

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self):
        # --- 此函数主体逻辑无需修改 ---
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            _, _, episode_reward = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print(f"total_steps:{self.total_steps} \t evaluate_reward:{evaluate_reward}")
        self.writer.add_scalar(f'evaluate_step_rewards_{self.env_name}', evaluate_reward, global_step=self.total_steps)
        np.save(f'./data_train/QMIX_env_{self.env_name}_number_{self.number}_seed_{self.seed}.npy', np.array(self.evaluate_rewards))
        
        if self.total_steps % 5e5 == 0 and self.total_steps > 0:
            print(f"Saving model at total_steps: {self.total_steps}")
            self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)


    def run_episode_mpe(self, evaluate=False):
        # --- 此函数主体逻辑无需修改 ---
        episode_buffer = {
            'obs_n': np.zeros([self.args.episode_limit, self.args.N, self.args.obs_dim]),
            's': np.zeros([self.args.episode_limit, self.args.state_dim]),
            'a_n': np.zeros([self.args.episode_limit, self.args.N]),
            'r_n': np.zeros([self.args.episode_limit, self.args.N]),
            'obs_next_n': np.zeros([self.args.episode_limit, self.args.N, self.args.obs_dim]),
            's_next': np.zeros([self.args.episode_limit, self.args.state_dim]),
            'done_n': np.ones([self.args.episode_limit, self.args.N])
        }
        
        episode_reward = 0
        observations, _ = self.env.reset()
        obs_n = np.array([observations[agent] for agent in observations.keys()])
        s = obs_n.flatten()
        
        hidden_state = np.zeros((self.args.N, self.args.rnn_hidden_dim))
        
        if not evaluate:
            self.epsilon = max(self.args.epsilon_finish, self.epsilon - (self.args.epsilon_start - self.args.epsilon_finish) / self.args.epsilon_anneal_time)
            self.writer.add_scalar('epsilon', self.epsilon, self.total_steps)
        
        current_epsilon = 0 if evaluate else self.epsilon

        for t in range(self.args.episode_limit):
            a_n, hidden_state_next = self.agent_n.choose_action(obs_n, hidden_state, current_epsilon, evaluate)

            actions_dict = {agent: a_n[i] for i, agent in enumerate(self.env.agents)}
            obs_next, r, done, _, _ = self.env.step(actions_dict)
            
            obs_next_n = np.array([obs_next[agent] for agent in obs_next.keys()])
            s_next = obs_next_n.flatten()
            r_n = np.array([r[agent] for agent in r.keys()])
            done_n = np.array([done[agent] for agent in done.keys()])
            
            episode_reward += r_n[0]

            if not evaluate:
                episode_buffer['obs_n'][t] = obs_n
                episode_buffer['s'][t] = s
                episode_buffer['a_n'][t] = a_n
                episode_buffer['r_n'][t] = r_n
                episode_buffer['obs_next_n'][t] = obs_next_n
                episode_buffer['s_next'][t] = s_next
                episode_buffer['done_n'][t] = done_n

            obs_n, s, hidden_state = obs_next_n, s_next, hidden_state_next
            
            if all(done_n):
                break
        
        return episode_buffer, t + 1, episode_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for QMIX in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=int, default=int(5e3), help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Evaluate times")
    parser.add_argument("--buffer_size", type=int, default=int(5e3), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The number of neurons in hidden layers of the qmix net")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--grad_norm_clip", type=float, default=10, help="Gradient norm clip")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon_finish", type=float, default=0.05, help="Final epsilon for exploration")
    parser.add_argument("--epsilon_anneal_time", type=int, default=int(2e6), help="How many steps to anneal epsilon")
    parser.add_argument("--target_update_freq", type=int, default=500, help="Steps to update target net")
    parser.add_argument('--render_mode', type=str, default='None', help='File path to my result')

    args = parser.parse_args()
    runner = Runner_QMIX_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    runner.run()