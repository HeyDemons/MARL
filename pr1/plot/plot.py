import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np


def moving_average(data, window_size=100):
    """简单移动平均"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def exponential_moving_average(data, alpha=0.1):
    """指数移动平均"""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def different_plot_rewards(csv_file, window_size=50, alpha=0.1,time=None):
    df = pd.read_csv(csv_file)

    # 计算平滑后的数据
    adv_ma = moving_average(df['Adversary Average Reward'].values, window_size)
    adv_ema = exponential_moving_average(df['Adversary Average Reward'].values, alpha)
    good_ema = exponential_moving_average(df['Good Average Reward'].values, alpha)
    good_ma = moving_average(df['Good Average Reward'].values, window_size)
    sum_ma = moving_average(df['Sum Reward of All Agents'].values, window_size)
    sum_ema = exponential_moving_average(df['Sum Reward of All Agents'].values, alpha)
    
    # 创建两个图形
    # 1. 移动平均对比图
    plt.figure(figsize=(15, 15))
    # 追捕者奖励
    plt.subplot(3, 1, 1)
    plt.plot(df['Episode'], df['Adversary Average Reward'], 'lightgray', alpha=0.3, label='Raw Data')
    plt.plot(df['Episode'][window_size-1:], adv_ma, 'b-', linewidth=2, label='Moving Average')
    plt.title('Adversary Average Reward - Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # 逃脱者奖励
    plt.subplot(3, 1, 2)
    plt.plot(df['Episode'], df['Good Average Reward'], 'lightgray', alpha=0.3, label='Raw Data')
    plt.plot(df['Episode'][window_size-1:], good_ma, 'g-', linewidth=2, label='Moving Average')
    plt.title('Good Average Reward - Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 总奖励
    plt.subplot(3, 1, 3)
    plt.plot(df['Episode'], df['Sum Reward of All Agents'], 'lightgray', alpha=0.3, label='Raw Data')
    plt.plot(df['Episode'][window_size-1:], sum_ma, 'b-', linewidth=2, label='Moving Average')
    plt.title('Sum Reward of All Agents - Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Sum Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存移动平均对比图
    save_path_ma = os.path.join(os.path.dirname(csv_file), f'rewards_plot_ma_{time}.png')
    plt.savefig(save_path_ma, dpi=300, bbox_inches='tight')
    
    # 2. 指数移动平均对比图
    plt.figure(figsize=(15, 15))
    # 追捕者奖励
    plt.subplot(3, 1, 1)
    plt.plot(df['Episode'], df['Adversary Average Reward'], 'lightgray', alpha=0.3, label='Raw Data')
    plt.plot(df['Episode'], adv_ema, 'r-', linewidth=2, label='Exponential Moving Average')
    plt.title('Adversary Average Reward - Exponential Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # 逃脱者奖励
    plt.subplot(3, 1, 2)
    plt.plot(df['Episode'], df['Good Average Reward'], 'lightgray', alpha=
0.3, label='Raw Data')
    plt.plot(df['Episode'], good_ema, 'g-', linewidth=2, label='Exponential Moving Average')
    plt.title('Good Average Reward - Exponential Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # 总奖励
    plt.subplot(3, 1, 3)
    plt.plot(df['Episode'], df['Sum Reward of All Agents'], 'lightgray', alpha=0.3, label='Raw Data')
    plt.plot(df['Episode'], sum_ema, 'r-', linewidth=2, label='Exponential Moving Average')
    plt.title('Sum Reward of All Agents - Exponential Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Sum Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存指数移动平均对比图
    save_path_ema = os.path.join(os.path.dirname(csv_file), f'rewards_plot_ema_{time}.png')
    plt.savefig(save_path_ema, dpi=300, bbox_inches='tight')
    
    print(f"Moving average plot saved to {save_path_ma}")
    print(f"Exponential moving average plot saved to {save_path_ema}")
    
    plt.show()

train_time = '2025-06-20_07-25'
if __name__ == "__main__":
    # CSV文件路径（相对于当前脚本的路径）
    csv_file = os.path.join(os.path.dirname(__file__), 'data', 'data_rewards_{}.csv'.format(train_time))
    print("csv_file name:",csv_file)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # print(df.head())
        # plot_rewards(csv_file)
        different_plot_rewards(csv_file, time = train_time)
    else:
        print(f"错误：未找到CSV文件：{csv_file}")