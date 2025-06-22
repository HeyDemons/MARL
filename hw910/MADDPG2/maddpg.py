import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from buffer import ReplayBuffer
from copy import deepcopy
from pettingzoo.mpe import simple_adversary_v3
import gymnasium as gym
import argparse
import time
import re
import pickle

def net_init(layer):
    fan_in = layer.weight.data.size(0)
    limit = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight, -limit, limit)
    nn.init.uniform_(layer.bias, -limit, limit)

def final_net_init(layer, low, high):
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, low, high)
        nn.init.uniform_(layer.bias, low, high)

class Actor(nn.Module): 
    def __init__(self, obs_dim, action_dim, hidden_dim = 128, supplement = None, pixel_case = False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        if supplement['net_init']:
            net_init(self.fc1)
            net_init(self.fc2)
            if pixel_case:
                final_net_init(self.fc3, -3e-4, 3e-4)
            else:
                final_net_init(self.fc3, -3e-3, 3e-3)
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action
    
class Critic(nn.Module):
    def __init__(self, dim_info:dict, hidden_dim = 128, supplement = None, pixel_case = False):
        super(Critic, self).__init__()
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        self.fc1 = nn.Linear(global_obs_act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        if supplement['net_init']:
            net_init(self.fc1)
            net_init(self.fc2)
            if pixel_case:
                final_net_init(self.fc3, -3e-4, 3e-4)
            else:
                final_net_init(self.fc3, -3e-3, 3e-3)
    #传入全局观测和动作
    def forward(self, obs, action):
        sa = torch.cat(list(obs) + list(action), dim = 1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
    
class DDPG:
    def __init__(self, obs_dim, action_dim, dim_info:dict, actor_lr, critic_lr, device, supplement):
        self.actor = Actor(obs_dim, action_dim, supplement = supplement).to(device)
        self.critic = Critic(dim_info, supplement = supplement).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        if supplement['weight_decay']:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr, weight_decay = 1e-3)
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

class MADDPG:
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick, supplement):
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = DDPG(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device, supplement)
            self.buffers[agent_id] = ReplayBuffer(buffer_size, obs_dim, act_dim= action_dim if is_continue else 1, device= device)
        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0]
        self.regular = False
        self.supplement = supplement
        if self.supplement['Batch_ObsNorm']:
            self.batch_size_obs_norm = {agent_id: Normalization_batch_size(dim_info[agent_id][0], device) for agent_id in dim_info.keys()}
        
    def select_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.supplement['Batch_ObsNorm']:
                obs = self.batch_size_obs_norm[agent_id](obs, update=False)
            if self.is_continue:
                action = self.agents[agent_id].actor(obs)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
            else:
                action = self.agents[agent_id].argmax(dim = 1).detach().cpu().numpy()[0]
                actions[agent_id] = action
        return actions
    
    def evaluate_action(self,obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            if self.is_continue: 
                action = self.agents[agent_id].actor(obs)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0)
        return actions
    
    def add(self, obs, action, reward, next_obs, done):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

    def sample(self, batch_size):
        total_size = len(self.buffers[self.agent_x])
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        next_action = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)
            if self.supplement['Batch_ObsNorm']:
                obs[agent_id] = self.batch_size_obs_norm[agent_id](obs[agent_id], update = True)
                next_obs[agent_id] = self.batch_size_obs_norm[agent_id](next_obs[agent_id], update = False)
            next_action[agent_id] = self.agents[agent_id].target_actor(next_obs[agent_id])

        return obs, action, reward, next_obs, done ,next_action
    
    def learn(self, batch_size, gamma, tau):
        for agent_id, agent in self.agents.items():

            obs, action, reward, next_obs, done, next_action = self.sample(batch_size)
            next_target_q = agent.target_critic(next_obs.values(), next_action.values())

            target_Q = reward[agent_id] + gamma * next_target_q * (1 - done[agent_id].float())
            current_Q = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            agent.update_critic(critic_loss)

            new_action = agent.actor(obs[agent_id])
            action[agent_id] = new_action
            actor_loss = -agent.critic(obs.values(), action.values()).mean()
            if self.regular:
                actor_loss += (new_action ** 2).mean() * 1e-3
            agent.update_actor(actor_loss)

        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for agent in self.agents.values():
            soft_update(agent.target_actor, agent.actor, tau)
            soft_update(agent.target_critic, agent.critic, tau)

    def save(self, model_path):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_path, 'MADDPG.pth')
        )
        
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue, model_dir,trick=None,supplement=None):
        policy = MADDPG(dim_info, is_continue = is_continue, actor_lr = 0, critic_lr = 0, buffer_size = 0, device = 'cpu',trick = trick,supplement = supplement)
        data = torch.load(os.path.join(model_dir, 'MADDPG.pth'))
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return policy


def get_env(env_name, env_agent_n = None):
    if env_name == 'simple_adversary_v3':
        env = simple_adversary_v3.parallel_env(max_cycles=25, continuous_actions=True, N = env_agent_n)
        env.reset()
    dim_info = {} # dict{agent_id:[obs_dim,action_dim]}
    for agent_id in env.agents:
        dim_info[agent_id] = []
        if isinstance(env.observation_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.observation_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(1)
        if isinstance(env.action_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.action_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(env.action_space(agent_id).n)
    return env, dim_info, 1, True

def make_dir(env_name,policy_name = 'DQN',trick = None):
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
    env_dir = os.path.join(script_dir,'./results', env_name)
    os.makedirs(env_dir) if not os.path.exists(env_dir) else None
    print('trick:',trick)
    # 确定前缀
    if trick is None or not any(trick.values()):
        prefix = policy_name + '_'
    else:
        prefix = policy_name + '_'
        for key in trick.keys():
            if trick[key]:
                prefix += key + '_'
    # 查找现有的文件夹并确定下一个编号
    pattern = re.compile(f'^{prefix}\d+') # ^ 表示开头，\d 表示数字，+表示至少一个
    existing_dirs = [d for d in os.listdir(env_dir) if pattern.match(d)]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.1, dt=1e-2, scale= None):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt 
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
        self.scale = scale 

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) +  np.sqrt(self.dt) * self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        if self.scale is None:
            return self.state 
        else:
            return self.state * self.scale

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False #是否更新均值和方差，在评估时，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RunningMeanStd_batch_size:
    # Dynamically calculate mean and std
    def __init__(self, shape,device):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape).to(device)
        self.S = torch.zeros(shape).to(device)
        self.std = torch.sqrt(self.S).to(device)

    def update(self, x):
        x = x.mean(dim=0,keepdim=True)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean 
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class Normalization_batch_size:
    def __init__(self, shape, device):
        self.running_ms = RunningMeanStd_batch_size(shape,device)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False #是否更新均值和方差，在评估时，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="simple_adversary_v3") 
    parser.add_argument("--N", type=int, default=2) # 环境中智能体数量 默认None 这里用来对比设置
    # 共有参数
    parser.add_argument("--seed", type=int, default=100) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(40000))
    parser.add_argument("--start_steps", type=int, default=500) # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=0)  # 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## AC参数
    parser.add_argument("--actor_lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=1e6) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=256)  #保证比start_steps小
    # DDPG 独有参数 
    ## gauss noise
    parser.add_argument("--gauss_sigma", type=float, default=1) # 高斯标准差 # 0.1 
    parser.add_argument("--gauss_scale", type=float, default=1)
    parser.add_argument("--gauss_init_scale", type=float, default=1) # 若不设置衰减，则设置成None 若设置了衰减 则 sigma = gauss_scale * gauss_sigma
    parser.add_argument("--gauss_final_scale", type=float, default=0)
    ## OU noise
    parser.add_argument("--ou_sigma", type=float, default=1) # 
    parser.add_argument("--ou_dt", type=float, default=1)
    parser.add_argument("--init_scale", type=float, default=1) # 若不设置衰减，则设置成None # maddpg值:ou_sigma 0.2 init_scale:0.3
    parser.add_argument("--final_scale", type=float, default=0.0) 
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MADDPG')
    parser.add_argument("--supplement", type=dict, default={'weight_decay':True,'OUNoise':True,'ObsNorm':False,'net_init':True,'Batch_ObsNorm':True}) # ObsNorm效果差于Batch_ObsNorm,选择Batch_ObsNorm
    parser.add_argument("--trick", type=dict, default=None)  
    # device参数   
    parser.add_argument("--device", type=str, default='cuda') # cpu/cuda
    
    args = parser.parse_args()
    if args.policy_name == 'MADDPG_simple' or args.supplement == {'weight_decay':False,'OUNoise':False,'ObsNorm':False,'net_init':False,'Batch_ObsNorm':False}:
        args.policy_name = 'MADDPG_simple'
        args.supplement = {'weight_decay':False,'OUNoise':False,'ObsNorm':False,'net_init':False,'Batch_ObsNorm':False}
    
    print(args)
    print('-' * 50)
    print('Algorithm:',args.policy_name)

    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name, env_agent_n = args.N)
    print(f'Env:{args.env_name}  dim_info:{dim_info}  max_action:{max_action}  max_episodes:{args.max_episodes}')
    
    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:',args.seed)

    ## 保存model文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name,trick=args.trick)
    writer = SummaryWriter(model_dir)
    print('model_dir:',model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    ## 算法配置
    policy = MADDPG(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick,args.supplement)

    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs,info = env.reset() # 改成env.reset()后增加其鲁棒性，但会导致seed失效
    {agent: env.action_space(agent).seed(seed = args.seed) for agent in env_agents}  # 针对action复现:env.action_space.sample()
    if args.gauss_init_scale is not None:
        args.gauss_scale = args.gauss_init_scale

    if args.supplement['ObsNorm']:
        obs_norm = {agent_id  :Normalization(shape = dim_info[agent_id][0]) for agent_id in env_agents }
        obs = {agent_id : obs_norm[agent_id](obs[agent_id]) for agent_id in env_agents }

    if args.supplement['OUNoise']:
        ou_noise = { agent_id : OUNoise(dim_info[agent_id][1], sigma = args.ou_sigma, dt=args.ou_dt, scale = args.init_scale) for agent_id in env_agents }#
    
    while episode_num < args.max_episodes:
        step +=1

        # 获取动作
        if step < args.random_steps: # # 区分环境里的action_ 和训练的action
            action_ = {agent: env.action_space(agent).sample() for agent in env_agents}  # [0,1]
            action = {agent_id: (action_[agent_id] * 2 - 1)* max_action for agent_id in env_agents} # [0,1] -> [-1,1]
        else:
            action = policy.select_action(obs) 
            # 加噪音  对于pettingzoo环境,得加上dtype = np.float32
            if args.supplement['OUNoise']:
                action_ = {agent_id: np.clip(action[agent_id] * max_action + ou_noise[agent_id].noise()* max_action, -max_action, max_action ,dtype = np.float32) for agent_id in env_agents}
            else:
                action_ = {agent_id: np.clip(action[agent_id] * max_action + args.gauss_scale * np.random.normal(scale = args.gauss_sigma * max_action, size = dim_info[agent_id][1]), -max_action, max_action ,dtype = np.float32) for agent_id in env_agents} 
            action_ = {agent_id: (action_[agent_id] + 1) / 2 for agent_id in env_agents} # [-1,1] -> [0,1] 
        
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        if args.supplement['ObsNorm']:
            next_obs = {agent_id : obs_norm[agent_id](next_obs[agent_id]) for agent_id in env_agents }

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id]  for agent_id in env_agents} ### truncated 为超过最大步数
        policy.add(obs, action, reward, next_obs, done_bool)
        for agent_id,r in reward.items():
            episode_reward[agent_id] += r

        obs = next_obs
        
        # episode 结束 ### 在pettingzoo中,env.agents 为空时  一个episode结束
        if any(done.values()):
            if args.supplement['OUNoise']:
                [ou_noise[agent_id].reset() for agent_id in env_agents]
                ## OUNoise scale若有 scale衰减 参考:https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/main.py#L71
                if args.init_scale is not None:
                    explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes # 剩余探索百分比
                    for agent_id in env_agents:
                        ou_noise[agent_id].scale = args.final_scale + (args.init_scale - args.final_scale) * explr_pct_remaining
            ## gauss_noise 衰减
            if args.gauss_init_scale is not None:
                explr_pct_remaining = max(0, args.max_episodes - (episode_num + 1)) / args.max_episodes
                args.gauss_scale = args.gauss_final_scale + (args.gauss_init_scale - args.gauss_final_scale) * explr_pct_remaining
            ## 显示
            if  (episode_num + 1) % 100 == 0 :
                print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
                for agent_id , r in episode_reward.items():
                    writer.add_scalar(f'reward_{agent_id}', r, episode_num + 1)
                    train_return[agent_id].append(r)
                                 
            episode_num += 1
            obs,info = env.reset() # 改成env.reset()后增加其鲁棒性，但会导致seed失效
            if args.supplement['ObsNorm']:
                obs = {agent_id : obs_norm[agent_id](obs[agent_id]) for agent_id in env_agents }
            episode_reward = {agent_id: 0 for agent_id in env_agents}
        
        # 满足step,更新网络
        if step > args.start_steps and step % args.learn_steps_interval == 0:
            policy.learn(args.batch_size, args.gamma, args.tau)

    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    if args.N is None:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),train_return_)
    else:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}_N_{len(env_agents)}.npy"),train_return_)
    
    if args.supplement['ObsNorm']:
        obs_norm_ = {agent_id: [obs_norm[agent_id].running_ms.mean, obs_norm[agent_id].running_ms.std] for agent_id in env_agents}
        pickle.dump(obs_norm_, open(os.path.join(model_dir,'obs_norm.pkl'), 'wb'))
    if args.supplement['Batch_ObsNorm']:
        batch_size_obs_norm_ = {agent_id: [policy.batch_size_obs_norm[agent_id].running_ms.mean.detach().cpu().numpy(), policy.batch_size_obs_norm[agent_id].running_ms.std.detach().cpu().numpy()] for agent_id in env_agents}
        pickle.dump(batch_size_obs_norm_, open(os.path.join(model_dir,'batch_size_obs_norm.pkl'), 'wb') )