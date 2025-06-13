import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import copy
from multiprocessing import Process, Pipe
from pettingzoo.mpe import simple_spread_v3
import datetime
import imageio

# Set a dummy video driver to avoid display errors in headless environments
os.environ['SDL_VIDEODRIVER'] = 'dummy'

class EnvWrapper:
    """A wrapper for the PettingZoo environment to handle dictionary-to-list conversions."""
    def __init__(self, render_mode=None):
        # Initialize the environment with a specified number of agents and max cycles
        self.env = simple_spread_v3.parallel_env(N=3, max_cycles=25, local_ratio=0.5, render_mode=render_mode)
        self.N = 3
        self.observation_space = self.env.observation_space('agent_0')
        self.action_space = self.env.action_space('agent_0')

    def reset(self, seed=None):
        """Resets the environment and returns observations and info as lists."""
        obs, infos = self.env.reset(seed=seed)
        return self.dict_to_list(obs)

    def step(self, action):
        """Takes a step in the environment and returns results as lists."""
        obs, rewards, dones, tructs, infos = self.env.step(self.list_to_dict(action))
        return self.dict_to_list(obs), self.dict_to_list(rewards), self.dict_to_list(dones), self.dict_to_list(tructs), self.dict_to_list(infos)
    
    def render(self):
        """Renders the environment."""
        return self.env.render()

    def list_to_dict(self, data):
        """Converts a list of agent data to a dictionary."""
        return {f"agent_{i}": data[i] for i in range(self.N)}

    def dict_to_list(self, data):
        """Converts a dictionary of agent data to a list."""
        return [data[f"agent_{i}"] for i in range(self.N)]

    def close(self):
        """Closes the environment."""
        self.env.close()

def worker(remote, parent_remote, env_fn_wrapper):
    """Function for a worker process to run an environment instance."""
    parent_remote.close()
    env = env_fn_wrapper()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, truct, info = env.step(data)
                remote.send((obs, reward, done, truct, info))
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError
        except EOFError:
            break


class ParallelEnv:
    """A class to manage multiple parallel environment instances."""
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, EnvWrapper))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def sample_action(self):
        """Samples a random action for each agent in each environment."""
        return np.random.randint(0, 5, size=(self.n_envs, 3))

    def step(self, actions):
        """Steps all parallel environments with the given actions."""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, tructs, infos = zip(*results)
        return obs, rewards, dones, tructs, infos

    def reset(self):
        """Resets all parallel environments."""
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        """Closes all parallel environments and worker processes."""
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


class Replay:
    """A simple replay buffer for storing and sampling transitions."""
    def __init__(self, min_size, max_size, batch_size):
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = []
        self.batch_size = batch_size

    def add(self, states, prev_actions, actions, rewards, next_states, dones):
        """Adds a batch of transitions to the buffer."""
        for i in range(len(rewards)):
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)
            self.buffer.append((states[i], prev_actions[i], actions[i], rewards[i], next_states[i], dones[i]))

    def sample(self):
        """Samples a batch of transitions from the buffer."""
        transitions = random.sample(self.buffer, self.batch_size)
        states, prev_actions, actions, rewards, next_states, dones = zip(*transitions)
        return states, prev_actions, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class RNN(nn.Module):
    """A simple RNN for the agent's Q-network."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        """Forward pass through the network."""
        x = self.norm1(F.relu(self.fc1(inputs)))
        x = self.norm2(F.relu(self.fc2(x)))
        return self.fc3(x)


class Qmix(nn.Module):
    """The Q-mixing network."""
    def __init__(self, state_dim, hidden_dim, num_agent):
        super(Qmix, self).__init__()
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_agent))
        self.hyper_b1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        # Use Softplus to ensure weights are non-negative
        self.trans_fn = nn.Softplus(beta=1, threshold=20)

    def forward(self, qs, states):
        """Forward pass for the mixing network."""
        weight = self.trans_fn(self.hyper_w1(states))
        bias = self.hyper_b1(states)
        return torch.sum(weight * qs, dim=-1, keepdim=True) + bias


class QmixAgent:
    """The main agent class that manages the networks, training, and actions."""
    def __init__(self, state_dim, action_dim, hidden_dim, num_agent, num_env, q_net_lr, qmix_lr, gamma, explore_rate, explore_rate_decay, min_explore_rate, update_gap, device):
        self.q_net = RNN(state_dim + action_dim, hidden_dim, action_dim).to(device)
        self.qmix = Qmix(state_dim * num_agent, hidden_dim, num_agent).to(device)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_qmix = copy.deepcopy(self.qmix)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=q_net_lr)
        self.qmix_optimizer = torch.optim.Adam(self.qmix.parameters(), lr=qmix_lr)
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agent = num_agent
        self.num_env = num_env
        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_rate_decay
        self.min_explore_rate = min_explore_rate
        self.device = device
        self.episode_len = 25
        self.gamma = gamma
        self.update_step = 0
        self.update_gap = update_gap

    def tdv(self, x):
        """Converts a numpy array to a torch tensor on the correct device."""
        return torch.tensor(np.array(x), dtype=torch.float32, device=self.device)

    def generate_inputs(self, states, prev_actions):
        """Generates the input for the RNN by concatenating states and one-hot encoded previous actions."""
        one_hot_action = F.one_hot(prev_actions.to(torch.long).squeeze(-1), num_classes=self.action_dim).to(torch.float32)
        return torch.cat([states, one_hot_action], dim=-1)

    def take_action(self, states, prev_action, explore=True):
        """Selects an action for each agent based on the current state and exploration rate."""
        with torch.no_grad():
            states = self.tdv(states).reshape(1, self.num_env * self.num_agent, -1)
            prev_action = self.tdv(prev_action).reshape(1, self.num_env * self.num_agent, -1)
            inputs = self.generate_inputs(states, prev_action)
            qs = self.q_net(inputs)
            
            if explore and np.random.rand() < self.explore_rate:
                return np.random.randint(0, self.action_dim, size=(self.num_env, self.num_agent))
            
            actions = qs.argmax(dim=-1).reshape(self.num_env, self.num_agent)
            return actions.cpu().numpy()

    def update(self, states, prev_actions, actions, rewards, next_states, dones):
        """Updates the network weights based on a batch of transitions."""
        bs = len(rewards)
        
        # Reshape and move data to the device
        states = self.tdv(states).transpose(0, 1).flatten(1, 2)
        prev_actions = self.tdv(prev_actions).unsqueeze(-1).transpose(0, 1).flatten(1, 2)
        actions = self.tdv(actions).unsqueeze(-1).transpose(0, 1).flatten(1, 2).to(torch.long)
        next_states = self.tdv(next_states).transpose(0, 1).flatten(1, 2)
        rewards = self.tdv(rewards).transpose(0, 1).sum(dim=-1, keepdim=True)
        dones = self.tdv(dones).transpose(0, 1)[:, :, :1]

        shared_states = states.reshape(self.episode_len, bs, -1)
        shared_next_states = next_states.reshape(self.episode_len, bs, -1)

        qs = self.q_net(self.generate_inputs(states, prev_actions))
        
        with torch.no_grad():
            next_qs_target = self.target_q_net(self.generate_inputs(next_states, actions))
            next_qs_selector = self.q_net(self.generate_inputs(next_states, actions))

        qs = qs.gather(-1, actions).reshape(self.episode_len, bs, self.num_agent)
        next_qs = next_qs_target.gather(-1, next_qs_selector.argmax(dim=-1, keepdim=True)).reshape(self.episode_len, bs, self.num_agent)

        values = self.qmix(qs, shared_states)
        with torch.no_grad():
            next_values = self.target_qmix(next_qs, shared_next_states)

        target_values = rewards + self.gamma * next_values * (1 - dones)
        loss = F.mse_loss(values, target_values.detach())
        
        self.q_net_optimizer.zero_grad()
        self.qmix_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        nn.utils.clip_grad_norm_(self.qmix.parameters(), 10.0)
        self.q_net_optimizer.step()
        self.qmix_optimizer.step()
        
        self.update_step += 1
        if self.update_step % self.update_gap == 0:
            self.copy_target()
            
        self.explore_rate = max(self.min_explore_rate, self.explore_rate * self.explore_rate_decay)
        return loss.item(), self.explore_rate

    def copy_target(self):
        """Copies weights from the main networks to the target networks."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_qmix.load_state_dict(self.qmix.state_dict())

    def save(self, path, identifier):
        """Saves the model weights with a specific identifier ('best' or 'final')."""
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.q_net.state_dict(), os.path.join(path, f'q_net_{identifier}.pth'))
        torch.save(self.qmix.state_dict(), os.path.join(path, f'qmix_{identifier}.pth'))

    def load(self, path, identifier):
        """Loads model weights with a specific identifier."""
        self.q_net.load_state_dict(torch.load(os.path.join(path, f'q_net_{identifier}.pth'), map_location=self.device))
        self.qmix.load_state_dict(torch.load(os.path.join(path, f'qmix_{identifier}.pth'), map_location=self.device))


def train_off_policy_agent(env, agent, replay, num_episodes, update_iter, log_dir):
    """The main training loop."""
    writer = SummaryWriter(log_dir)
    return_list = []
    i_eps = 0
    loss = 0.0
    best_avg_return = -np.inf
    model_dir = os.path.join(log_dir, 'models')

    with tqdm(total=num_episodes, desc='Training Episodes') as pbar:
        for i_episode in range(num_episodes):
            episode_return = 0
            state = env.reset()
            prev_action = env.sample_action()
            
            transition_dict = {'states': [], 'prev_actions': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            
            for _ in range(agent.episode_len):
                action = agent.take_action(state, prev_action)
                next_state, reward, _, truct, _ = env.step(action)
                
                transition_dict['states'].append(state)
                transition_dict['prev_actions'].append(prev_action)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(truct)
                
                state = next_state
                prev_action = action
                episode_return += np.array(reward).sum() / len(reward)

            for key, value in transition_dict.items():
                transition_dict[key] = np.array(value).swapaxes(0, 1)
            
            replay.add(transition_dict['states'], transition_dict['prev_actions'], transition_dict['actions'], transition_dict['rewards'], transition_dict['next_states'], transition_dict['dones'])
            
            if len(replay) >= replay.min_size:
                for _ in range(update_iter):
                    states, prev_actions, actions, rewards, next_states, dones = replay.sample()
                    loss, explore_rate = agent.update(states, prev_actions, actions, rewards, next_states, dones)
                
                writer.add_scalar('Metrics/Loss', loss, i_eps)
                writer.add_scalar('Metrics/Explore Rate', explore_rate, i_eps)
                writer.add_scalar('Rewards/Individual Reward', np.array(transition_dict['rewards']).mean(), i_eps)

            return_list.append(episode_return)
            writer.add_scalar('Rewards/Episode Return', episode_return, i_eps)
            i_eps += 1
            
            current_avg_return = np.mean(return_list[-100:]) if len(return_list) > 0 else 0.0
            
            if len(return_list) > 100 and current_avg_return > best_avg_return:
                best_avg_return = current_avg_return
                agent.save(model_dir, 'best')
                writer.add_scalar('Rewards/Best Average Return', best_avg_return, i_eps)

            pbar.set_postfix({'episode': i_eps,
                              'avg_return': f'{current_avg_return:.3f}',
                              'explore': f'{agent.explore_rate:.3f}',
                              'loss': f'{loss:.3f}'})
            pbar.update(1)
    
    agent.save(model_dir, 'final')
    env.close()
    return return_list, model_dir

def evaluate_and_record(agent_params, model_path, model_identifier, video_path):
    """Evaluates the trained agent and records a video of its performance."""
    print(f"Starting evaluation for model: {model_identifier}")
    eval_env = EnvWrapper(render_mode='rgb_array')
    agent = QmixAgent(**agent_params)
    agent.load(model_path, model_identifier)

    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    
    # Specify codec explicitly
    video_writer = imageio.get_writer(video_path, fps=10, codec='libx264')

    state = eval_env.reset()
    state = [np.array(s) for s in state]
    prev_action = [[np.random.randint(0, agent.action_dim) for _ in range(agent.num_agent)]]
    
    total_reward = 0

    for _ in range(25):
        frame = eval_env.render()
        video_writer.append_data(frame)
        
        action_batch = agent.take_action([state], prev_action, explore=False)
        action = action_batch[0]

        next_state, reward, _, _, _ = eval_env.step(action)
        
        state = [np.array(s) for s in next_state]
        prev_action = [action]
        total_reward += np.sum(reward)
    
    video_writer.close()
    eval_env.close()
    print(f"Evaluation finished. Total reward: {total_reward}")
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    # --- Hyperparameters ---
    min_size = 500
    max_size = 100000
    batch_size = 128
    
    state_dim = 18
    action_dim = 5
    hidden_dim = 128
    num_agent = 3
    num_env = 20
    
    q_net_lr = 5e-4
    qmix_lr = 5e-4
    gamma = 0.99
    explore_rate = 1.0
    explore_rate_decay = 0.99995
    min_explore_rate = 0.01
    update_gap = 200
    
    device_name = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device_name}")

    num_episodes = 100000 # Using total episodes directly
    update_iter = 1

    # --- Setup ---
    # torch.manual_seed(1)
    # np.random.seed(1)
    # if device_name == "cuda:1":
    #     torch.cuda.manual_seed_all(1)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', 'qmix_experiment_' + current_time)
    print(f"Logs will be saved to: {log_dir}")

    replay = Replay(min_size, max_size, batch_size)
    env = ParallelEnv(num_env)
    
    agent_params = {
        'state_dim': state_dim, 'action_dim': action_dim, 'hidden_dim': hidden_dim,
        'num_agent': num_agent, 'num_env': num_env, 'q_net_lr': q_net_lr,
        'qmix_lr': qmix_lr, 'gamma': gamma, 'explore_rate': explore_rate,
        'explore_rate_decay': explore_rate_decay, 'min_explore_rate': min_explore_rate,
        'update_gap': update_gap, 'device': device
    }
    agent = QmixAgent(**agent_params)

    # --- Training ---
    return_list, model_path = train_off_policy_agent(env, agent, replay, num_episodes, update_iter, log_dir)

    # --- Evaluation and Video Recording ---
    eval_agent_params = agent_params.copy()
    eval_agent_params['num_env'] = 1
    
    # Evaluate the BEST model
    print("\n--- Evaluating Best Model ---")
    video_path_best = os.path.join(log_dir, 'evaluation_video_best.mp4')
    evaluate_and_record(eval_agent_params, model_path, 'best', video_path_best)
    
    # Evaluate the FINAL model
    print("\n--- Evaluating Final Model ---")
    video_path_final = os.path.join(log_dir, 'evaluation_video_final.mp4')
    evaluate_and_record(eval_agent_params, model_path, 'final', video_path_final)

