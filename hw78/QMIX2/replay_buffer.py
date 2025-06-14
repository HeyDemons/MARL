import numpy as np

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.buffer_size = args.buffer_size
        self.current_size = 0
        self.counter = 0

        self.buffer = {
            'obs_n': np.empty([self.buffer_size, self.episode_limit, self.N, self.obs_dim]),
            's': np.empty([self.buffer_size, self.episode_limit, self.state_dim]),
            'a_n': np.empty([self.buffer_size, self.episode_limit, self.N, 1]),
            'r_n': np.empty([self.buffer_size, self.episode_limit, self.N]),
            'obs_next_n': np.empty([self.buffer_size, self.episode_limit, self.N, self.obs_dim]),
            's_next': np.empty([self.buffer_size, self.episode_limit, self.state_dim]),
            'done_n': np.empty([self.buffer_size, self.episode_limit, self.N])
        }

    def store_episode(self, episode_data):
        idx = self.counter % self.buffer_size
        for key in self.buffer.keys():
            self.buffer[key][idx] = episode_data[key]
        
        self.counter += 1
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        batch = {}
        for key in self.buffer.keys():
            batch[key] = self.buffer[key][indices]
        return batch