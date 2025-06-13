import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
import os
import copy
from multiprocessing import Process, Pipe
from tqdm import tqdm
import wandb
import imageio
from pettingzoo.mpe import simple_adversary_v3

# ================================================================= #
#                             1. Configuration                        #
# ================================================================= #

config = {
    # Replay Buffer
    "min_size": 2000,
    "max_size": 1000000,
    "batch_size": 1024,
    # Environment
    "state_dim": 10,
    "action_dim": 2,
    "hidden_dim": 128,
    "num_agent": 3,
    "num_env": 20,
    # Training
    "actor_lr": 5e-4,
    "critic_lr": 5e-4,
    "gamma": 0.99,
    "explore_rate": 1.3,
    "explore_rate_decay": 0.99995,
    "min_explore_rate": 0.01,
    "update_gap": 100,  # Target network update frequency
    "num_episodes": 100000, # Reduced for faster example run, original was 2_000_000 // num_env
    "update_iter": 1,
    # System
    "device": torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
    "seed": 1
}


# ================================================================= #
#                      2. Environment Utilities                       #
# ================================================================= #

class EnvWrapper:
    """A wrapper for the PettingZoo environment to handle observation processing."""
    def __init__(self, render_mode=None):
        self.padding = np.zeros(2)
        # Initialize the environment with the specified render mode
        self.env = simple_adversary_v3.parallel_env(render_mode=render_mode, continuous_actions=True)

    def process_obs(self, obs):
        """Pads the adversary's observation to match the agent's observation dimension."""
        obs['adversary_0'] = np.concatenate([self.padding, obs['adversary_0']], axis=0)
        return obs

    def reset(self, seed=None):
        obs, infos = self.env.reset(seed=seed)
        obs = self.process_obs(obs)
        return self.dict_to_list(obs), self.dict_to_list(infos)

    def step(self, action):
        """Converts agent actions to the format expected by the environment and steps."""
        action = action.astype(np.float32)
        full_action = np.zeros((3, 5), dtype=np.float32)
        full_action[:, 1] = -action[:, 0]
        full_action[:, 2] = action[:, 0]
        full_action[:, 3] = -action[:, 1]
        full_action[:, 4] = action[:, 1]
        full_action = np.where(full_action < 0, 0, full_action)
        
        obs, rewards, dones, tructs, infos = self.env.step(self.list_to_dict(full_action))
        obs = self.process_obs(obs)
        return (self.dict_to_list(obs), self.dict_to_list(rewards),
                self.dict_to_list(dones), self.dict_to_list(tructs),
                self.dict_to_list(infos))

    def render(self):
        """Renders the environment and returns the frame."""
        return self.env.render()

    def list_to_dict(self, infos):
        return {'adversary_0': infos[0], 'agent_0': infos[1], 'agent_1': infos[2]}

    def dict_to_list(self, infos):
        return [infos['adversary_0'], infos['agent_0'], infos['agent_1']]

def worker(remote, parent_remote, env_fn):
    """Worker process for parallel environments."""
    parent_remote.close()
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'reset':
                remote.send(env.reset()[0])
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
        except EOFError:
            break

class ParallelEnv:
    """Manages multiple environments running in parallel processes."""
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        env_fn = lambda: EnvWrapper(render_mode=None)
        self.ps = [Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, tructs, infos = zip(*results)
        return obs, rewards, dones, tructs, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                pass
        for p in self.ps:
            p.join()


# ================================================================= #
#                         3. Replay Buffer                          #
# ================================================================= #

class Replay:
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


# ================================================================= #
#                         4. Model Definitions                        #
# ================================================================= #

class PolicyNet(nn.Module):
    """Actor Network."""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        mu = self.mu(x)
        std = F.softplus(self.std(x))
        dist = Normal(mu, std)
        normal_action = dist.rsample()
        log_prob = dist.log_prob(normal_action)
        action = torch.tanh(normal_action)
        # Enforcing Action Bounds
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob

class ValueNet(nn.Module):
    """Critic Network."""
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc(x)


# ================================================================= #
#                         5. SAC Agent                              #
# ================================================================= #

class SAC:
    """Single-agent SAC implementation."""
    def __init__(self, state_dim, hidden_dim, action_dim, agent_id, num_agent, num_env, actor_lr, critic_lr, gamma, explore_rate, explore_rate_decay, min_explore_rate, update_gap, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic1 = ValueNet((state_dim + action_dim) * num_agent, hidden_dim).to(device)
        self.critic2 = ValueNet((state_dim + action_dim) * num_agent, hidden_dim).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.device = device
        self.gamma = gamma
        self.agent_id = agent_id
        self.update_step = 0
        self.update_gap = update_gap
        self.num_env = num_env
        self.action_dim = action_dim
        self.explore_rate = explore_rate
        self.explore_rate_decay = explore_rate_decay
        self.min_explore_rate = min_explore_rate
        self.entropy_coef = 0.01

    def take_action(self, states, deterministic=False):
        if not deterministic and np.random.rand() < self.explore_rate:
            return np.random.uniform(-1, 1, size=(self.num_env, self.action_dim))
        
        with torch.no_grad():
            actions, _ = self.actor(states)
            if deterministic:
                return actions.cpu().numpy()
            return actions.cpu().numpy()

    def update(self, states, shared_states, actions, online_actions, rewards, shared_next_states, online_next_actions, online_next_log_probs, dones):
        shared_input = torch.cat([shared_states, actions], -1)
        qs1 = self.critic1(shared_input)
        qs2 = self.critic2(shared_input)

        with torch.no_grad():
            shared_next_inputs = torch.cat([shared_next_states, online_next_actions], -1)
            next_qs1 = self.target_critic1(shared_next_inputs)
            next_qs2 = self.target_critic2(shared_next_inputs)
            next_entropy = -online_next_log_probs[self.agent_id].mean(dim=-1, keepdim=True)
            soft_next_qs = torch.min(next_qs1, next_qs2) + self.entropy_coef * next_entropy
            target_qs = rewards + self.gamma * soft_next_qs * (1 - dones)
        
        critic_loss1 = F.mse_loss(qs1, target_qs)
        critic_loss2 = F.mse_loss(qs2, target_qs)

        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        action, log_prob = self.actor(states)
        online_actions_copy = list(online_actions)
        online_actions_copy[self.agent_id] = action
        online_actions_cat = torch.cat(online_actions_copy, -1)
        
        shared_inputs_actor = torch.cat([shared_states, online_actions_cat], dim=-1)
        qs1_actor = self.critic1(shared_inputs_actor)
        qs2_actor = self.critic2(shared_inputs_actor)
        
        entropy = -log_prob.mean(dim=-1, keepdim=True)
        soft_qs = torch.min(qs1_actor, qs2_actor) + self.entropy_coef * entropy
        actor_loss = -soft_qs.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_step += 1
        if self.update_step % self.update_gap == 0:
            self.copy_target()
        
        self.explore_rate = max(self.min_explore_rate, self.explore_rate * self.explore_rate_decay)
        
        return actor_loss.item(), critic_loss1.item(), critic_loss2.item()

    def copy_target(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def save(self, prefix):
        os.makedirs("models", exist_ok=True)
        torch.save(self.actor.state_dict(), f"models/{prefix}_actor.pth")
        torch.save(self.critic1.state_dict(), f"models/{prefix}_critic1.pth")

    def load(self, prefix):
        self.actor.load_state_dict(torch.load(f"models/{prefix}_actor.pth", map_location=self.device))
        self.critic1.load_state_dict(torch.load(f"models/{prefix}_critic1.pth", map_location=self.device))
        self.copy_target()


class MASAC:
    """Multi-Agent SAC Controller."""
    def __init__(self, state_dim, hidden_dim, action_dim, num_agent, num_env, actor_lr, critic_lr, gamma, explore_rate, explore_rate_decay, min_explore_rate, update_gap, device):
        self.agents = [SAC(state_dim, hidden_dim, action_dim, i, num_agent, num_env, actor_lr, critic_lr, gamma, explore_rate, explore_rate_decay, min_explore_rate, update_gap, device) for i in range(num_agent)]
        self.num_agent = num_agent
        self.device = device

    def tdv(self, x):
        return torch.tensor(np.array(x), dtype=torch.float32).to(self.device)

    def save(self, i):
        for j in range(self.num_agent):
            self.agents[j].save(f"{i}_{j}")

    def load(self, i):
        for j in range(self.num_agent):
            # Adversary (agent 0) is not trained, so we don't load its model.
            if j > 0:
                self.agents[j].load(f"{i}_{j}")
    
    def take_action(self, states, deterministic=False):
        states_tensor = self.tdv(states)
        actions = []
        for i in range(self.num_agent):
            # For agent 0 (adversary), we can use random actions during evaluation if not trained
            if deterministic and i == 0:
                action = np.random.uniform(-1, 1, size=(states_tensor.shape[0], config['action_dim']))
            else:
                action = self.agents[i].take_action(states_tensor[:, i, :], deterministic)
            actions.append(action)
        return np.array(actions).swapaxes(0, 1)

    def update(self, states, actions, rewards, next_states, dones, global_step):
        bs = len(rewards)
        states_tensor = self.tdv(states)
        next_states_tensor = self.tdv(next_states)
        shared_states = states_tensor.reshape(bs, -1)
        shared_next_states = next_states_tensor.reshape(bs, -1)
        actions_tensor = self.tdv(actions).reshape(bs, -1)
        rewards_tensor = self.tdv(rewards).unsqueeze(-1)
        dones_tensor = self.tdv(dones).unsqueeze(-1)

        online_actions, online_next_actions, online_next_log_probs = [], [], []
        with torch.no_grad():
            for i in range(self.num_agent):
                _, online_log_prob = self.agents[i].actor(states_tensor[:, i, :])
                online_next_action, online_next_log_prob = self.agents[i].actor(next_states_tensor[:, i, :])
                online_actions.append(online_log_prob)
                online_next_actions.append(online_next_action)
                online_next_log_probs.append(online_next_log_prob)
        
        online_next_actions_cat = torch.cat(online_next_actions, -1)
        
        log_dict = {}
        # We only train the "good" agents (agents 1 and 2), not the adversary (agent 0)
        for i in range(1, self.num_agent):
            actor_loss, critic1_loss, critic2_loss = self.agents[i].update(
                states_tensor[:, i, :], shared_states, actions_tensor, online_actions,
                rewards_tensor[:, i, :], shared_next_states, online_next_actions_cat,
                online_next_log_probs, dones_tensor[:, i, :]
            )
            log_dict[f'agent_{i}/actor_loss'] = actor_loss
            log_dict[f'agent_{i}/critic1_loss'] = critic1_loss
            log_dict[f'agent_{i}/critic2_loss'] = critic2_loss
            log_dict[f'agent_{i}/explore_rate'] = self.agents[i].explore_rate
        
        wandb.log(log_dict, step=global_step)


# ================================================================= #
#                       6. Training and Evaluation                    #
# ================================================================= #

def train_off_policy_agent(env, agent, replay, cfg):
    """Main training loop."""
    return_list = []
    adversary_return_list = []
    global_step = 0
    
    for i in range(10): # Outer loop for checkpointing
        with tqdm(total=int(cfg['num_episodes'] / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(cfg['num_episodes'] / 10)):
                episode_return = np.zeros(cfg['num_agent'])
                state = env.reset()
                done = False
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, dones, tructs, _ = env.step(action)
                    replay.add(state, action, reward, next_state, tructs)
                    state = next_state
                    episode_return += np.array(reward).mean(axis=0)
                    done = any(d[0] for d in dones) or any(t[0] for t in tructs)
                    global_step += 1

                if len(replay) >= cfg['min_size']:
                    for _ in range(cfg['update_iter']):
                        states, actions, rewards, next_states, dones_sample = replay.sample()
                        agent.update(states, actions, rewards, next_states, dones_sample, global_step)
                
                return_list.append(episode_return[1:].sum())
                adversary_return_list.append(episode_return[0])
                
                # Log rewards to wandb
                wandb.log({
                    "reward/good_agents_return": episode_return[1:].sum(),
                    "reward/adversary_return": episode_return[0],
                    "global_step": global_step
                })

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f"{(cfg['num_episodes'] / 10 * i + i_episode + 1):.0f}",
                        'return': f"{np.mean(return_list[-10:]):.3f}",
                        'adv_return': f"{np.mean(adversary_return_list[-10:]):.3f}"
                    })
                pbar.update(1)
        
        # Save model checkpoint
        agent.save(i)
        
    
    return 9 # Return the index of the last saved model

def evaluate_and_save_gif(agent, cfg, model_idx):
    """Evaluates the agent and saves a GIF of the episode."""
    print("Starting evaluation and GIF creation...")
    agent.load(model_idx)
    
    # Use a single, non-parallel environment for rendering
    eval_env = EnvWrapper(render_mode="rgb_array")
    state, _ = eval_env.reset(seed=cfg['seed'])
    
    frames = []
    done = False
    
    for i in range(25):  # Limit to 1000 steps for evaluation
        # Add a dimension for the single environment
        state_for_agent = np.expand_dims(state, axis=0)  # 修复形状问题
        frames.append(eval_env.render())
        
        action = agent.take_action(state_for_agent, deterministic=True)
        # Remove the extra dimension from the action
        next_state, _, dones, tructs, _ = eval_env.step(action[0])
        
        state = next_state
        done = any(dones) or any(tructs)
        if done:
            break
    eval_env.env.close()
    
    # Save GIF
    gif_path = "evaluation.gif"
    # fps=30 转换为 duration=33 (1000 / 30)
    imageio.mimsave(gif_path, frames, duration=33)
    print(f"Evaluation GIF saved to {gif_path}")
    
    # Log GIF to wandb
    wandb.log({"evaluation_video": wandb.Video(gif_path, fps=30, format="gif")})


# ================================================================= #
#                         7. Main Execution                         #
# ================================================================= #

if __name__ == "__main__":
    # --- 1. Initialization ---
    wandb.init(project="MPE_Simple_Adversary_SAC", config=config, name=f"sac_run_{config['seed']}")
    
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 2. Setup for Training ---
    # Hide video output during training for performance
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    replay_buffer = Replay(config['min_size'], config['max_size'], config['batch_size'])
    env = ParallelEnv(config['num_env'])
    agent = MASAC(
        state_dim=config['state_dim'], hidden_dim=config['hidden_dim'], action_dim=config['action_dim'],
        num_agent=config['num_agent'], num_env=config['num_env'], actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'], gamma=config['gamma'], explore_rate=config['explore_rate'],
        explore_rate_decay=config['explore_rate_decay'], min_explore_rate=config['min_explore_rate'],
        update_gap=config['update_gap'], device=config['device']
    )
    
    # --- 3. Start Training ---
    print("Starting training...")
    last_model_idx = train_off_policy_agent(env, agent, replay_buffer, config)
    env.close()
    
    # --- 4. Start Evaluation ---
    evaluate_and_save_gif(agent, config, last_model_idx)

    wandb.finish()
    print("Run finished.")
