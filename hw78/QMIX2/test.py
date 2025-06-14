import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from train import Runner_QMIX_MPE # 
from PIL import Image

# 默认路径设置
reward_path_default = './data_train/'
result_path_default = './result_qmix/' # 为 QMIX 创建一个新的结果文件夹
if not os.path.exists(result_path_default):
    os.makedirs(result_path_default)

parser = argparse.ArgumentParser("Hyperparameters Setting for QMIX in MPE environment")
# 从 train_qmix.py 复制与环境和模型结构相关的参数
parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")

# 测试脚本的特定参数
parser.add_argument('--model_load_step', type=int, default=1000, help="The step number of the model to load (in K, e.g., 1000 for 1000k steps)")
parser.add_argument('--num_eval_episodes', type=int, default=5, help="Number of episodes to run for evaluation and GIF generation")
parser.add_argument('--reward_path', type=str,
                    default=os.path.join(reward_path_default, 'QMIX_env_simple_spread_v3_number_1_seed_0.npy'),
                    help='File path to the reward file')
parser.add_argument('--result_path', type=str,
                    default=result_path_default, help='File path to save results (GIF, plot)')
parser.add_argument('--render_mode', type=str,
                    default='rgb_array', help='Render mode for the environment')

# 解析命令行参数
args = parser.parse_args()
result_dir = args.result_path

# --- GIF 保存设置 ---
assert os.path.exists(result_dir)
gif_dir = os.path.join(result_dir, 'gif')
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)
gif_num = len([file for file in os.listdir(gif_dir)]) # 当前GIF数量，用于命名新文件

# --- 初始化 Runner 和加载模型 ---
# 注意：这里我们实例化 Runner_QMIX_MPE
runner = Runner_QMIX_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
# 加载QMIX模型 (只加载Agent的RNN网络，因为QMixer在评估时不需要，且Agent网络决定了行为)
runner.agent_n.load_model("simple_spread_v3", number=1, seed=0, step=args.model_load_step)

print(f"开始评估，将运行 {args.num_eval_episodes} 个回合...")

# --- 运行评估回合并生成GIF ---
for episode in range(args.num_eval_episodes):
    observations, _ = runner.env.reset()
    # 在每个回合开始时重置QMIX代理的隐藏状态
    runner.agent_n.eval_hidden = None 
    
    agent_reward = {agent: 0 for agent in runner.env.agents}  # 当前回合的智能体奖励
    frame_list = []  # 用于保存GIF的帧

    for episode_step in range(runner.args.episode_limit):
        obs_n = np.array([observations[agent] for agent in observations.keys()])
        
        # 使用 QMIX 代理选择动作，evaluate=True 表示贪婪选择（不使用epsilon-greedy）
        a_n = runner.agent_n.choose_action(obs_n, evaluate=True)
        
        # 将动作数组转换为环境所需的字典格式
        actions = {agent: a_n[i] for i, agent in enumerate(runner.env.agents)}

        next_observations, rewards, dones, _, _ = runner.env.step(actions)
        
        # 仅在第一个评估回合中渲染并保存帧以制作GIF
        if episode == 0:
            frame_list.append(Image.fromarray(runner.env.render()))
            
        observations = next_observations

        for agent_id, reward in rewards.items():  # 更新奖励
            agent_reward[agent_id] += reward
        
        if all(dones.values()):
            break

    message = f'回合 {episode + 1}, '
    for agent_id, reward in agent_reward.items():
        message += f'{agent_id}: {reward:>4f}; '
    print(message)
    
    # 在第一个回合结束后保存GIF
    if episode == 0 and frame_list:
        gif_path = os.path.join(gif_dir, f'out_qmix_{gif_num + 1}.gif')
        frame_list[0].save(gif_path, save_all=True, append_images=frame_list[1:], duration=100, loop=0)
        print(f"GIF 已保存至: {gif_path}")

runner.env.close()

# --- 绘制奖励曲线 ---
print(f"从 {args.reward_path} 加载奖励数据并绘图...")
try:
    rewards = np.load(args.reward_path)
    fig, ax = plt.subplots()
    x = range(1, np.shape(rewards)[0] + 1)
    ax.plot(x, rewards, label='Average Episode Reward')
    ax.legend()
    ax.set_xlabel('Evaluation Cycle')
    ax.set_ylabel('Reward')
    title = 'Evaluate Result of QMIX'
    ax.set_title(title)
    
    # 保存图像
    fig_path = os.path.join(result_dir, 'reward_qmix.png')
    fig.savefig(fig_path, dpi=300)
    print(f"奖励曲线图已保存至: {fig_path}")
    plt.show()

except FileNotFoundError:
    print(f"错误: 找不到奖励文件 at {args.reward_path}。请检查路径或先运行训练。")