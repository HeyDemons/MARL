import numpy as np

class ReplayBuffer_QMIX:
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        
        # 为每个 transition 元素创建存储空间
        self.buffer = {'obs_n': np.zeros([self.buffer_size, args.episode_limit, args.N, args.obs_dim]),
                       's': np.zeros([self.buffer_size, args.episode_limit, args.state_dim]),
                       'a_n': np.zeros([self.buffer_size, args.episode_limit, args.N]),
                       'r_n': np.zeros([self.buffer_size, args.episode_limit, args.N]),
                       'obs_next_n': np.zeros([self.buffer_size, args.episode_limit, args.N, args.obs_dim]),
                       's_next': np.zeros([self.buffer_size, args.episode_limit, args.state_dim]),
                       'done_n': np.ones([self.buffer_size, args.episode_limit, args.N]),
                       'actual_len': np.zeros([self.buffer_size, 1]) # 新增：存储 episode 实际长度
                       }

    def store_episode(self, episode_data, episode_actual_length): # 修改：增加 episode_actual_length 参数
        """
        存储一个完整的 episode 数据。
        episode_data 是一个包含所有 transition 数据的字典。
        """
        idx = self.count % self.buffer_size
        for key in episode_data.keys(): # 只迭代 episode_data 中的键，避免 'actual_len' 不存在于 episode_data 时出错
            self.buffer[key][idx] = episode_data[key]
        self.buffer['actual_len'][idx] = episode_actual_length # 存储实际长度
        
        self.count += 1
        self.current_size = min(self.count, self.buffer_size)

    def sample(self):
        """
        从 buffer 中随机采样一个 batch 的 episodes。
        """
        if self.current_size < self.batch_size:
            return None # 如果数据不足，则不采样
        
        # 随机选择 batch_size 个索引
        indices = np.random.choice(self.current_size, self.batch_size, replace=False)
        
        batch = {}
        for key in self.buffer.keys():
            batch[key] = self.buffer[key][indices]
            
        return batch