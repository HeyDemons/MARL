import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Agent's Q-network
class Agent_RNN(nn.Module):
    def __init__(self, args):
        super(Agent_RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.obs_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        q = self.fc2(h_out)
        return q, h_out

# Mixing Network
class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()
        self.args = args
        
        # Hypernetwork for weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(args.state_dim, args.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.qmix_hidden_dim, args.N * args.qmix_hidden_dim)
        )
        # Hypernetwork for bias
        self.hyper_b1 = nn.Linear(args.state_dim, args.qmix_hidden_dim)

        # Second layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(args.state_dim, args.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.qmix_hidden_dim, args.qmix_hidden_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(args.state_dim, args.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.qmix_hidden_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs shape: (batch_size, N)
        # states shape: (batch_size, state_dim)
        
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.args.N) # Reshape for batch matrix multiplication
        
        # First layer
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.args.N, self.args.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)
        
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        # Compute final Q_tot
        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(batch_size, -1, 1)
        
        return q_total

class QMIX_MPE:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        
        # Epsilon-greedy exploration
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay = (self.args.epsilon - self.min_epsilon) / self.args.max_train_steps

        # Networks
        self.eval_agent_rnn = Agent_RNN(args)
        self.target_agent_rnn = Agent_RNN(args)
        self.eval_qmixer = QMixer(args)
        self.target_qmixer = QMixer(args)

        # Copy initial weights
        self.target_agent_rnn.load_state_dict(self.eval_agent_rnn.state_dict())
        self.target_qmixer.load_state_dict(self.eval_qmixer.state_dict())
        
        # Optimizer
        self.agent_params = list(self.eval_agent_rnn.parameters()) + list(self.eval_qmixer.parameters())
        self.optimizer = torch.optim.Adam(params=self.agent_params, lr=self.args.lr)

        # Hidden states
        self.eval_hidden = None
        self.target_hidden = None

    def choose_action(self, obs_n, evaluate=False):
        self.eval_hidden = self.eval_hidden if self.eval_hidden is not None else torch.zeros((self.N, self.args.rnn_hidden_dim))
        
        obs_tensor = torch.tensor(obs_n, dtype=torch.float32)
        q_values, self.eval_hidden = self.eval_agent_rnn(obs_tensor, self.eval_hidden)
        
        # Epsilon-greedy
        if not evaluate and np.random.uniform() < self.epsilon:
            actions = np.random.randint(0, self.action_dim, self.N)
        else:
            actions = q_values.argmax(dim=1).numpy()
            
        return actions
        
    def train(self, replay_buffer, total_steps):
        if replay_buffer.current_size < self.args.batch_size:
            return

        batch = replay_buffer.sample(self.args.batch_size)
        
        # Unpack batch
        obs_n = torch.tensor(batch['obs_n'], dtype=torch.float32)
        s = torch.tensor(batch['s'], dtype=torch.float32)
        a_n = torch.tensor(batch['a_n'], dtype=torch.long)
        r_n = torch.tensor(batch['r_n'], dtype=torch.float32)
        obs_next_n = torch.tensor(batch['obs_next_n'], dtype=torch.float32)
        s_next = torch.tensor(batch['s_next'], dtype=torch.float32)
        done_n = torch.tensor(batch['done_n'], dtype=torch.float32)
        
        # Reshape for training
        batch_size = self.args.batch_size
        obs_n = obs_n.view(batch_size, self.args.episode_limit, self.N, -1)
        s = s.view(batch_size, self.args.episode_limit, -1)
        a_n = a_n.view(batch_size, self.args.episode_limit, self.N, -1)
        r_n = r_n.view(batch_size, self.args.episode_limit, self.N)
        obs_next_n = obs_next_n.view(batch_size, self.args.episode_limit, self.N, -1)
        s_next = s_next.view(batch_size, self.args.episode_limit, -1)
        done_n = done_n.view(batch_size, self.args.episode_limit, self.N)

        q_evals, q_targets = [], []
        eval_hidden = torch.zeros((batch_size, self.N, self.args.rnn_hidden_dim))
        target_hidden = torch.zeros((batch_size, self.N, self.args.rnn_hidden_dim))

        for t in range(self.args.episode_limit):
            obs_t = obs_n[:, t]
            s_t = s[:, t]
            a_t = a_n[:, t]
            r_t = r_n[:, t].mean(dim=1, keepdim=True) # Use mean reward
            obs_next_t = obs_next_n[:, t]
            s_next_t = s_next[:, t]
            done_t = done_n[:, t].mean(dim=1, keepdim=True)

            q_eval_t, eval_hidden = self.eval_agent_rnn(obs_t.reshape(-1, self.obs_dim), eval_hidden.reshape(-1, self.args.rnn_hidden_dim))
            q_eval_t = q_eval_t.view(batch_size, self.N, -1)
            q_eval_t_action = q_eval_t.gather(2, a_t).squeeze(2)

            q_target_t, target_hidden = self.target_agent_rnn(obs_next_t.reshape(-1, self.obs_dim), target_hidden.reshape(-1, self.args.rnn_hidden_dim))
            q_target_t = q_target_t.view(batch_size, self.N, -1)
            
            q_evals.append(self.eval_qmixer(q_eval_t_action, s_t))
            q_targets.append(self.target_qmixer(q_target_t.max(dim=2)[0], s_next_t))

        q_evals = torch.stack(q_evals, dim=1).squeeze()
        q_targets = torch.stack(q_targets, dim=1).squeeze()

        # TD Target
        y = r_n.mean(dim=-1) + self.args.gamma * (1 - done_n.mean(dim=-1)) * q_targets
        
        loss = F.mse_loss(q_evals, y.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_clip_norm)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        
        # Update target networks
        if total_steps % self.args.target_update_freq == 0:
            self.target_agent_rnn.load_state_dict(self.eval_agent_rnn.state_dict())
            self.target_qmixer.load_state_dict(self.eval_qmixer.state_dict())
            
    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.eval_agent_rnn.state_dict(), "./model/QMIX_agent_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.eval_agent_rnn.load_state_dict(torch.load("./model/QMIX_agent_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))