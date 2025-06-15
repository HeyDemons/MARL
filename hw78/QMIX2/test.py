import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from train import Runner_QMIX_MPE # Assuming Runner_QMIX_MPE is in train.py
from PIL import Image

reward_data_path = './data_train/'
results_output_path = './results_qmix/' # Changed to avoid conflict with MAPPO results

if not os.path.exists(results_output_path):
    os.makedirs(results_output_path)

def main():
    parser = argparse.ArgumentParser("Hyperparameters Setting for QMIX Evaluation in MPE environment")
    # Arguments needed by Runner_QMIX_MPE and QMIX_MPE constructors
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The number of neurons in hidden layers of the qmix net")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (required by QMIX_MPE, not used in eval)")
    # Evaluation specific arguments
    parser.add_argument("--num_eval_episodes", type=int, default=5, help="Number of episodes to run for evaluation")
    parser.add_argument("--render_mode", type=str, default='rgb_array', help="Render mode for environment (rgb_array for GIF)")
    parser.add_argument("--model_env_name", type=str, default="simple_spread_v3", help="Environment name for model loading")
    parser.add_argument("--model_number", type=int, default=1, help="Model number for loading")
    parser.add_argument("--model_seed", type=int, default=0, help="Model seed for loading")
    parser.add_argument("--model_load_step_k", type=int, default=3000, help="Model checkpoint step in thousands (e.g., 3000 for 3M steps)")
    
    parser.add_argument('--trained_reward_path', type=str,
                        default=os.path.join(reward_data_path, 'QMIX_env_simple_spread_v3_number_1_seed_0.npy'),
                        help='File path to QMIX training rewards file for plotting')
    parser.add_argument('--results_dir', type=str,
                        default=results_output_path, help='Directory to save GIFs and plots')

    # Dummy args that might be expected by Runner_QMIX_MPE or QMIX_MPE from the training script's argparser
    # These may not be strictly necessary if Runner_QMIX_MPE/QMIX_MPE handles their absence gracefully for eval
    parser.add_argument("--max_train_steps", type=int, default=int(3e6))
    parser.add_argument("--evaluate_freq", type=int, default=int(5e3))
    parser.add_argument("--evaluate_times", type=int, default=3)
    parser.add_argument("--buffer_size", type=int, default=int(5e3))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grad_norm_clip", type=float, default=10)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_finish", type=float, default=0.05)
    parser.add_argument("--epsilon_anneal_time", type=int, default=int(1e6))
    parser.add_argument("--target_update_freq", type=int, default=200)


    args = parser.parse_args()

    # Ensure the results directory and GIF subdirectory exist
    gif_dir = os.path.join(args.results_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    
    # Initialize runner
    # The 'number' and 'seed' for Runner_QMIX_MPE are for its internal logging/saving setup during training.
    # For evaluation, we primarily care about loading the correct model specified by model_args.
    runner = Runner_QMIX_MPE(args, env_name=args.model_env_name, number=args.model_number, seed=args.model_seed)

    # Load pre-trained model
    try:
        runner.agent_n.load_model(args.model_env_name, args.model_number, args.model_seed, args.model_load_step_k)
    except FileNotFoundError as e:
        print(e)
        return # Exit if model loading fails

    # --- Evaluation Loop ---
    total_episode_rewards = []
    gif_num_base = len([file for file in os.listdir(gif_dir) if file.startswith('qmix_eval_')])

    for episode_idx in range(args.num_eval_episodes):
        print(f"Running evaluation episode {episode_idx + 1}/{args.num_eval_episodes}")
        observations, _ = runner.env.reset()
        
        # Initial observation processing
        obs_n = np.array([observations[agent] for agent in runner.env.agents])
        
        # Initialize hidden state for QMIX DRQN
        # Access N and rnn_hidden_dim from runner.args, which are set after env creation
        hidden_state = np.zeros((runner.args.N, args.rnn_hidden_dim), dtype=np.float32)

        current_episode_reward = 0
        frame_list = []

        for episode_step in range(args.episode_limit):
            actions_int, hidden_state_next = runner.agent_n.choose_action(obs_n, hidden_state, epsilon=0, evaluate=True)
            
            actions_dict = {agent: actions_int[i] for i, agent in enumerate(runner.env.agents)}
            
            obs_next_dict, rewards_dict, dones_dict, _, _ = runner.env.step(actions_dict)
            
            obs_next_n = np.array([obs_next_dict[agent] for agent in runner.env.agents])
            
            if args.render_mode == 'rgb_array':
                frame = runner.env.render()
                if frame is not None:
                    frame_list.append(Image.fromarray(frame))
            
            # Sum of rewards for all agents at this step (or use a specific agent's reward if preferred)
            step_reward = sum(rewards_dict.values())
            current_episode_reward += step_reward
            
            obs_n = obs_next_n
            hidden_state = hidden_state_next # Update hidden state

            if all(dones_dict.values()):
                break
        
        total_episode_rewards.append(current_episode_reward)
        print(f"Episode {episode_idx + 1} finished. Reward: {current_episode_reward:.2f}")

        # Save GIF for the episode
        if frame_list:
            try:
                frame_list[0].save(os.path.join(gif_dir, f'qmix_eval_{gif_num_base + episode_idx + 1}.gif'),
                                   save_all=True, append_images=frame_list[1:], duration=100, loop=0)
                print(f"Saved GIF: qmix_eval_{gif_num_base + episode_idx + 1}.gif")
            except IndexError:
                print("Warning: No frames to save for GIF.")
            except Exception as e:
                print(f"Error saving GIF: {e}")
        else:
            print("Warning: frame_list is empty, cannot save GIF. Ensure render_mode is 'rgb_array' and env.render() is working.")


    avg_eval_reward = np.mean(total_episode_rewards)
    print(f"\nAverage evaluation reward over {args.num_eval_episodes} episodes: {avg_eval_reward:.2f}")

    # --- Plotting Training Rewards ---
    if os.path.exists(args.trained_reward_path):
        try:
            training_rewards = np.load(args.trained_reward_path)
            fig, ax = plt.subplots()
            x_values = range(1, len(training_rewards) + 1)
            ax.plot(x_values, training_rewards, label='QMIX Training Rewards')
            ax.legend()
            ax.set_xlabel('Training Episodes/Evaluation Points')
            ax.set_ylabel('Reward')
            title = f'QMIX Training Rewards ({os.path.basename(args.trained_reward_path)})'
            ax.set_title(title)
            plot_save_path = os.path.join(args.results_dir, 'qmix_training_rewards.png')
            fig.savefig(plot_save_path, dpi=300)
            print(f"Saved training rewards plot to: {plot_save_path}")
            # plt.show() # Optionally display the plot
        except Exception as e:
            print(f"Error loading or plotting training rewards: {e}")
    else:
        print(f"Warning: Training reward file not found at {args.trained_reward_path}")

    runner.env.close()
    print("Evaluation finished.")

if __name__ == '__main__':
    main()