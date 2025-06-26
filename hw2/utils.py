import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
from IPython.display import clear_output
import os

def plot_training_results(dqn_scores=None, pg_scores=None, window=10):
    """Plot the training progress of DQN and/or Policy Gradient algorithms."""
    plt.figure(figsize=(10, 6), dpi=300)
    
    if dqn_scores is not None:
        # Plot raw scores
        plt.plot(dqn_scores, alpha=0.3, color='blue', label='DQN Raw Scores')
        
        # Plot moving average
        if len(dqn_scores) >= window:
            dqn_moving_avg = np.convolve(dqn_scores, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(dqn_scores)), dqn_moving_avg, color='blue', label=f'DQN Moving Avg (window={window})')

    if pg_scores is not None:
        # Plot raw scores
        plt.plot(pg_scores, alpha=0.3, color='red', label='Policy Gradient Raw Scores')
        
        # Plot moving average
        if len(pg_scores) >= window:
            pg_moving_avg = np.convolve(pg_scores, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(pg_scores)), pg_moving_avg, color='red', label=f'Policy Gradient Moving Avg (window={window})')

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

def evaluate_agent(agent, env_name='CartPole-v1', num_episodes=10, max_steps=1000, render=False):
    """Evaluate a trained agent's performance."""
    env = gym.make(env_name, render_mode='human' if render else None)
    scores = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_steps):
            if hasattr(agent, 'select_action'):
                # For both DQN and Policy Gradient
                action = agent.select_action(state, training=False) if 'training' in agent.select_action.__code__.co_varnames else agent.select_action(state)
            else:
                # Fallback
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = torch.argmax(agent.policy_net(state_tensor)).item()
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            state = next_state
            
            if done:
                break
                
        scores.append(score)
        
    avg_score = np.mean(scores)
    print(f"Evaluation over {num_episodes} episodes: Avg Score = {avg_score:.2f}")
    return avg_score

def record_video(agent, env_name='CartPole-v1', video_length=500, algorithm_name="agent"):
    """Record a video of the agent's performance."""
    try:
        import gym.wrappers
        import gymnasium as gym
        
        # Create algorithm-specific directory for videos
        video_dir = f'./videos/{algorithm_name}/'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        # Create environment with video recording
        env = gym.make(env_name, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True, name_prefix=algorithm_name)
        
        state, _ = env.reset()
        for t in range(video_length):
            if hasattr(agent, 'select_action'):
                # For both DQN and Policy Gradient
                action = agent.select_action(state, training=False) if 'training' in agent.select_action.__code__.co_varnames else agent.select_action(state)
            else:
                # Fallback
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = torch.argmax(agent.policy_net(state_tensor)).item()
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            
            if done:
                break
                
        env.close()
        print(f"Video recorded and saved to {video_dir}")
    except Exception as e:
        print(f"Failed to record video: {e}")

def plot_episode_progress(episode, score, max_episodes, window=100, scores=None):
    """Plot the progress of training during an episode."""
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.title(f'Episode {episode}/{max_episodes}')
    
    if scores:
        plt.plot(scores)
        plt.axhline(y=500, color='r', linestyle='-', alpha=0.3)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(0, 500, 'Goal', color='r')
        if len(scores) > window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(scores)), moving_avg, color='orange')
            
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()