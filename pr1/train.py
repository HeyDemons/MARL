from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3, simple_spread_v3
from agent.maddpg.maddpg_agent import MADDPG
from agent.masac.masac_agent import MASAC
from agent.matd3.matd3_agent import MATD3
from agent.mappo.mappo_agent import MAPPO
import torch
import os
import time
from datetime import timedelta
from utils.parameters import parameters
from utils.logger import TrainingLogger
from utils.runner import RUNNER, MAPPO_RUNNER

def get_env(env_name, ep_len=25, render_mode ="None"):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array",continuous_actions=True)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode=render_mode, num_good=1, num_adversaries=3, num_obstacles=0, max_cycles=ep_len, continuous_actions=True)
    new_env.reset(seed = 0)
    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:",agent_id)
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = [] #[low action,  hign action]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)
    print("_dim_info:",_dim_info)
    print("action_bound:",action_bound)
    return new_env, _dim_info, action_bound

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    start_time = time.time()
    args = parameters()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, "models", args.env_name)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    print("using Env's name:", args.env_name)
    env, dim_info, action_bound = get_env(args.env_name, ep_len=args.episode_length, render_mode=args.render_mode)
    print("dim_info:", dim_info)
    if args.algorithm == 'MADDPG':
        print("using algorithm: MADDPG")
        agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device)
    elif args.algorithm == 'MASAC':
        print("using algorithm: MASAC")
        alpha_lr = args.alpha_lr if args.alpha_lr is not None else 2e-4
        agent = MASAC(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, alpha_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device)  
    elif args.algorithm == 'MATD3':
        print("using algorithm: MATD3")
        agent = MATD3(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device) 
    elif args.algorithm == 'MAPPO':
        print("using alogrthim: MAPPO")
        agent = MAPPO(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}. Please choose from 'MADDPG', 'MASAC', 'MAPPO' or 'MATD3'.")
    # 创建运行对象
    if args.algorithm == 'MAPPO':
        runner = MAPPO_RUNNER(agent, env, args, device, mode = 'train')
    else:
        runner = RUNNER(agent, env, args, device, mode = 'train')
    runner.train()
    end_time = time.time()
    training_time = end_time - start_time
    training_time_str = str(timedelta(seconds=int(training_time)))
    print(f"\n===========训练完成!===========")
    print(f"训练设备: {device}")
    print(f"训练用时: {training_time_str}")


    logger = TrainingLogger()
    current_time = logger.save_training_log(args, device, training_time_str, runner)
    print(f"完成时间: {current_time}")

    print("--- saving trained models ---")
    agent.save_model()
    print("--- trained models saved ---")