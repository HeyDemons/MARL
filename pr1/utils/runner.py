import numpy as np
import visdom
import csv
import os
from datetime import datetime
import tqdm as tqdm
class RUNNER:
    def __init__(self, agent, env, par, device, mode = 'evaluate'):
        self.agent = agent
        self.env = env
        self.par = par
        # 这里为什么新建而不是直接使用用agent.agents.keys()？
        # 因为pettingzoo中智能体死亡，这个字典就没有了，会导致 td target更新出错。所以这里维护一个不变的字典。
        self.env_agents = [agent_id for agent_id in self.agent.agents.keys()]
        self.done = {agent_id: False for agent_id in self.agent.agents.keys()}

        # 添加奖励记录相关的属性
        self.reward_sum_record = []  # 用于平滑的奖励记录
        self.all_reward_record = []  # 保存所有奖励记录，用于最终统计
        self.all_adversary_avg_rewards = []  # 追捕者平均奖励
        self.all_good_agent_avg_rewards = []  # 逃跑者平均奖励
        self.all_sum_rewards = []  # 所有智能体总奖励
        self.episode_rewards = {}  # 每个智能体的奖励历史

        # 将 agent 的模型放到指定设备上
        if par.algorithm == 'MADDPG':
            for agent in self.agent.agents.values():
                agent.actor.to(device)
                agent.target_actor.to(device)
                agent.critic.to(device)
                agent.target_critic.to(device)
        elif par.algorithm == 'MASAC':
            for agent in self.agent.agents.values():
                agent.actor.to(device)
                agent.critic_1.to(device)
                agent.critic_2.to(device)
                agent.target_critic_1.to(device)
                agent.target_critic_2.to(device)       


        if mode == 'train' and self.par.visdom:
            self.viz = visdom.Visdom()
            self.viz.close()
        else: # evaluate模式下不需要visdom
            pass


    def train(self):
        step = 0
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.par.episode_num) for agent_id in self.env.agents}
        epsilon_decay_episodes = int(0.8 * self.par.episode_num)  
        epsilon_decay = (self.par.epsilon_start - self.par.epsilon_min) / epsilon_decay_episodes  
        epsilon = self.par.epsilon_start # 初始epsilon值
        
        # 创建tqdm进度条
        pbar = tqdm.tqdm(range(self.par.episode_num), desc='Training Progress', ncols=120)
        
        # episode循环
        for episode in pbar:
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset()
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env_agents}
            # 每个智能体与环境进行交互
            while self.env.agents:  #  加入围捕判断
                step += 1
                # epsilon采样
                if np.random.rand() < epsilon:
                    # 随机选择动作
                    action = {agent_id: self.env.action_space(agent_id).sample() for agent_id in self.env.agents}
                # 开始学习 根据策略选择动作
                else:
                    action = self.agent.select_action(obs)  
                # 执行动作 获得下一状态 奖励 终止情况
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}

                # 存储样本
                self.agent.add(obs, action, reward, next_obs, self.done)
                # 计算当前episode每个智能体的奖励 每个step求和
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                # 开始学习 有学习开始条件 有学习频率
                if step >= self.par.random_steps and step % self.par.learn_interval == 0:
                    # # 学习
                    self.agent.learn(self.par.batch_size, self.par.gamma)
                    # 更新网络
                if step >= self.par.random_steps and step % self.par.update_interval == 0:
                    # 更新目标网络
                    self.agent.update_target(self.par.tau)
                # 状态更新
                obs = next_obs

            # 计算当前episode的总奖励和平均奖励
            sum_reward = sum(agent_reward.values())
            
            # 计算adversary和good agent的平均奖励
            adversary_rewards_list = [r for agent_id, r in agent_reward.items() if agent_id.startswith('adversary_')]
            good_agent_rewards_list = [r for agent_id, r in agent_reward.items() if agent_id.startswith('agent_')]
            
            avg_adversary_reward = np.mean(adversary_rewards_list) if adversary_rewards_list else 0
            avg_good_agent_reward = np.mean(good_agent_rewards_list) if good_agent_rewards_list else 0
            
            # 更新epsilon值
            if episode < epsilon_decay_episodes:
                epsilon = max(self.par.epsilon_min, epsilon - epsilon_decay)
            else:
                epsilon = self.par.epsilon_min
            
            # 更新tqdm的后缀信息
            pbar.set_postfix({
                'Eps': f'{epsilon:.3f}',
                'Adv_Avg': f'{avg_adversary_reward:.2f}',
                'Good_Avg': f'{avg_good_agent_reward:.2f}'
            })
                
            # 记录当前episode围捕者的平均奖励
            self.all_adversary_avg_rewards.append(avg_adversary_reward)
            self.all_good_agent_avg_rewards.append(avg_good_agent_reward)
            
            # 记录当前episode的所有智能体和奖励 存储到csv中
            self.all_sum_rewards.append(sum_reward)
            # 记录当前episode的所有智能体和奖励 为奖励平滑做准备
            self.reward_sum_record.append(sum_reward)

            self.all_reward_record.append(sum_reward)  # 保存完整记录
            # 保存当前智能体在当前episode的奖励
            for agent_id, r in agent_reward.items():
                self.episode_rewards[agent_id][episode] = r  #  episode_rewards  字典： {agent_id:[episoed1_reward, episode2_reward,...]}
            
            # 根据平滑窗口确定打印间隔 并进行平滑
            if (episode + 1) % self.par.size_win == 0:  #  500 步平滑一次
                message = f'episode {episode + 1}, '
                sum_reward_check = 0
                for agent_id, r in agent_reward.items():
                    message += f'{agent_id}: {r:>4f}; ' # r:>4f 是格式化字符串，用于保留四位小数。
                    sum_reward_check += r
                message += f'sum reward: {sum_reward_check}'
                #换行print
                print('\n'+ message)
                self.reward_sum_record = []

        # 保存数据到文件（CSV格式）
        self.save_rewards_to_csv(self.all_adversary_avg_rewards, self.all_good_agent_avg_rewards, self.all_sum_rewards)

    def get_running_reward(self, arr):

        if len(arr) == 0:  # 如果传入空数组，使用完整记录
            arr = self.all_reward_record

        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        window = self.par.size_win
        running_reward = np.zeros_like(arr)

        for i in range(len(arr)):
            # 对每个i，确保窗口大小不会超出数组的实际大小
            start_idx = max(0, i - window + 1)
            running_reward[i] = np.mean(arr[start_idx:i + 1])
        # print(f"running_reward{running_reward}")
        return running_reward

    @staticmethod
    def exponential_moving_average(rewards, alpha=0.1):
        """计算指数移动平均奖励"""
        ema_rewards = np.zeros_like(rewards)
        ema_rewards[0] = rewards[0]
        for t in range(1, len(rewards)):
            ema_rewards[t] = alpha * rewards[t] + (1 - alpha) * ema_rewards[t - 1]
        return ema_rewards

    def moving_average(self, rewards):
        """计算简单移动平均奖励"""
        window_size = self.par.size_win
        sma_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        return sma_rewards
    
    """保存围捕者平均奖励和所有智能体总奖励到 CSV 文件"""
    def save_rewards_to_csv(self, adversary_rewards, good_rewards, sum_rewards, filename = None): # filename="data_rewards.csv"
        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        if filename is None:
            filename = f"data_rewards_{timestamp}.csv"
        # 获取 runner.py 所在目录，并生成与 utils 同级的 plot 目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件（runner.py）的绝对路径
        plot_dir = os.path.join(current_dir, '..', 'plot', 'data')  # 获取与 utils 同级的 plot 文件夹
        os.makedirs(plot_dir, exist_ok=True)  # 创建 plot 目录（如果不存在）

        # 构造完整的 CSV 文件路径
        full_filename = os.path.join(plot_dir, filename)

        header = ['Episode', 'Adversary Average Reward', 'Good Average Reward', 'Sum Reward of All Agents']
        data = list(zip(range(1, len(adversary_rewards) + 1), adversary_rewards, good_rewards, sum_rewards))
        # 将数据写入 CSV 文件
        with open(full_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # 写入表头
            writer.writerows(data)  # 写入数据

        print(f"Rewards data saved to {full_filename}")
#============================================================================================================

    def evaluate(self):
        # # 使用visdom实时查看训练曲线
        # viz = None
        # if self.par.visdom:
        #     viz = visdom.Visdom()
        #     viz.close()
        # step = 0
        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        self.reward_sum_record = []
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.par.episode_num) for agent_id in self.env.agents}
        # episode循环
        for episode in range(self.par.episode_num):
            step = 0  # 每回合step重置
            print(f"评估第 {episode + 1} 回合")
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset()  # 重置环境，开始新回合
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            # 每个智能体与环境进行交互
            while self.env.agents:
                # print(f"While num:{step}")
                step += 1
                # 使用训练好的智能体选择动作（没有随机探索）
                action = self.agent.select_action(obs)
                # 执行动作 获得下一状态 奖励 终止情况
                # 下一状态：字典 键为智能体名字 值为对应的下一状态
                # 奖励：字典 键为智能体名字 值为对应的奖励
                # 终止情况：bool
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}

                # 累积每个智能体的奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                obs = next_obs

                
                if step % 10 == 0:
                    print(f"Step {step}, obs: {obs}, action: {action}, reward: {reward}, done: {self.done}")

            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)


class MAPPO_RUNNER:
    def __init__(self, agent, env, par, device, mode='train'):
        self.agent = agent
        self.env = env
        self.par = par
        self.env_agents = list(self.agent.agents.keys())
        
        # Reward recording attributes
        self.reward_sum_record = []
        self.all_reward_record = []
        self.all_adversary_avg_rewards = []
        self.all_sum_rewards = []
        
        if mode == 'train' and self.par.visdom:
            self.viz = visdom.Visdom()
            self.viz.close()

    def train(self):
        for episode in tqdm.tqdm(range(self.par.episode_num), desc='Training Progress', ncols=100):
            obs, _ = self.env.reset()
            agent_reward = {agent_id: 0 for agent_id in self.env_agents}

            # This loop is the data collection phase (rollout)
            while self.env.agents:
                # Select action using the current policy
                action, log_probs, values = self.agent.select_action(obs)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                done = {agent_id: bool(terminated.get(agent_id, False) or truncated.get(agent_id, False)) for agent_id in self.env_agents}

                # Store the full experience tuple in the on-policy buffer
                self.agent.add(obs, action, reward, done, log_probs, values)

                for agent_id, r in reward.items():
                    # Check if agent_id from reward dict is valid
                    if agent_id in agent_reward:
                        agent_reward[agent_id] += r
                
                obs = next_obs

            # --- End of Rollout ---

            # The learning step happens ONCE at the end of the episode
            if episode >= self.par.learn_start_episode:
                self.agent.learn()

            # --- Logging and Visualization ---
            sum_reward = sum(agent_reward.values())
            self.log_rewards(episode, agent_reward, sum_reward)

        self.save_rewards_to_csv(self.all_adversary_avg_rewards, self.all_sum_rewards)

    def evaluate(self):
        for episode in range(self.par.episode_num):
            print(f"评估第 {episode + 1} 回合")
            obs, _ = self.env.reset()
            
            while self.env.agents:
                # For evaluation, we only need the action
                action, _, _ = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                obs = next_obs
        print("Evaluation finished.")

    def log_rewards(self, episode, agent_reward, sum_reward):
        """Helper function for logging and plotting rewards."""
        self.all_sum_rewards.append(sum_reward)
        self.reward_sum_record.append(sum_reward)

        adversary_rewards_list = [r for agent_id, r in agent_reward.items() if agent_id.startswith('adversary_')]
        avg_adversary_reward = np.mean(adversary_rewards_list) if adversary_rewards_list else 0
        self.all_adversary_avg_rewards.append(avg_adversary_reward)

        if self.par.visdom:
            for agent_id, r in agent_reward.items():
                self.viz.line(X=[episode + 1], Y=[r], win=f'reward_{agent_id}', opts={'title': f'Reward of {agent_id}'}, update='append')
            self.viz.line(X=[episode + 1], Y=[sum_reward], win='Sum Reward', opts={'title': 'Sum Reward of All Agents'}, update='append')
            self.viz.line(X=[episode + 1], Y=[avg_adversary_reward], win='Adversary Avg Reward', opts={'title': 'Average Reward of Adversaries'}, update='append')

        if (episode + 1) % self.par.size_win == 0:
            avg_win_reward = np.mean(self.reward_sum_record)
            print(f'Episode {episode + 1}, Average Sum Reward (last {self.par.size_win}): {avg_win_reward:.2f}')
            self.reward_sum_record = []

    def save_rewards_to_csv(self, adversary_rewards, sum_rewards, filename=None):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        if filename is None:
            filename = f"mappo_rewards_{timestamp}.csv"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(current_dir, '..', 'plot', 'data')
        os.makedirs(plot_dir, exist_ok=True)
        full_filename = os.path.join(plot_dir, filename)

        header = ['Episode', 'Adversary Average Reward', 'Sum Reward of All Agents']
        data = list(zip(range(1, len(adversary_rewards) + 1), adversary_rewards, sum_rewards))
        
        with open(full_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data)
        print(f"Rewards data saved to {full_filename}")