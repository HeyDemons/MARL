import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import math
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import FrameStack, GrayScaleObservation

ENV_NAME = "CarRacing-v2"
SEED = 42


DISCRETE_ACTIONS_LIST = [
    np.array([0.0, 1.0, 0.0]),    # Action 0: 直行, 全油门 (快速前进)
    np.array([-0.6, 0.5, 0.0]),   # Action 1: 中等左转, 中等油门 (适用于一般弯道)
    np.array([0.6, 0.5, 0.0]),    # Action 2: 中等右转, 中等油门 (适用于一般弯道)
    np.array([-1.0, 0.2, 0.0]),   # Action 3: 大幅度左转, 轻油门 (适用于急转弯，保持动力)
    np.array([1.0, 0.2, 0.0]),    # Action 4: 大幅度右转, 轻油门 (适用于急转弯，保持动力)
    np.array([0.0, 0.0, 0.8]),    # Action 5: 直行, 强制刹车
    np.array([-0.3, 0.3, 0.1]),   # Action 6: 轻微左转, 轻油门, 轻刹车 (精细控制，减速入弯或修正)
    np.array([0.3, 0.3, 0.1]),    # Action 7: 轻微右转, 轻油门, 轻刹车 (精细控制，减速入弯或修正)
    np.array([0.0, 0.3, 0.0])     # Action 8: 直行, 轻油门 (缓慢加速或保持低速)
]
NUM_ACTIONS = len(DISCRETE_ACTIONS_LIST)

#DQN Parameters
FRAME_STACK_SIZE = 4
INPUT_CHANNELS = FRAME_STACK_SIZE 
IMG_HEIGHT, IMG_WIDTH = 96, 96 

BUFFER_SIZE = 50000        # Replay buffer capacity
BATCH_SIZE = 32            # Minibatch size for training
GAMMA = 0.99               # Discount factor for future rewards
EPS_START = 1.0            # Starting value of epsilon for epsilon-greedy
EPS_END = 0.05             # Minimum value of epsilon
EPS_DECAY_FRAMES = 100000  # Number of frames over which to decay epsilon
TARGET_UPDATE_FREQ = 1000  # How often to update target network (in agent steps)
LEARNING_RATE = 1e-4       # Learning rate for the optimizer

NUM_EPISODES_TRAIN = 500   # Number of episodes for training (CarRacing needs many more for good performance)
MAX_STEPS_PER_EPISODE = 1000 # Max steps per episode for CarRacing-v2

# Paths
MODEL_DIR = "dqn_models_pytorch"
MODEL_NAME = "dqn_carracing_pytorch.pth"
VIDEO_DIR = "dqn_videos_pytorch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class DQN(nn.Module):
    def __init__(self, input_channels, height, width, n_actions):
        super(DQN, self).__init__()
        # Input shape: (batch_size, input_channels, height, width)
        # Example: (batch_size, 4, 96, 96) for 4 stacked 96x96 grayscale frames

        self.conv_stack = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), # Output: (N, 32, 23, 23) for 96x96 input
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),             # Output: (N, 64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),             # Output: (N, 64, 8, 8)
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the flattened size after convolutions
        # Create a dummy input to pass through conv_stack to get the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            flattened_size = self.conv_stack(dummy_input).shape[1]

        self.fc_stack = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions) # Output Q-values for each action
        )

    def forward(self, x):
        # Normalize pixel values if not already done (states from FrameStack are uint8)
        x = x.float() / 255.0
        x = self.conv_stack(x)
        return self.fc_stack(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # States from FrameStack are LazyFrames, convert to numpy array before storing
        state = np.array(state)
        next_state = np.array(next_state)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # Unzip transitions
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32), # Ensure rewards are float
                np.array(next_states),
                np.array(dones, dtype=np.uint8)) # Ensure dones are int/bool

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_channels, height, width, n_actions, device,
                 lr=LEARNING_RATE, gamma=GAMMA, buffer_size=BUFFER_SIZE,
                 target_update_freq=TARGET_UPDATE_FREQ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.policy_net = DQN(input_channels, height, width, n_actions).to(device)
        self.target_net = DQN(input_channels, height, width, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Initialize target_net with policy_net weights
        self.target_net.eval()  # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0 # For epsilon decay and target network updates

    def select_action(self, state, epsilon):
        """Selects an action using an epsilon-greedy policy."""
        if random.random() > epsilon:
            with torch.no_grad(): # No need to track gradients for action selection
                # Convert state (LazyFrames or np.array) to tensor for the network
                state_np = np.array(state)
                state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device) # Add batch dim
                # state_tensor already normalized in DQN.forward
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item() # Get action with max Q-value
        else:
            action = random.randrange(self.n_actions)
        self.steps_done += 1
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize_model(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None # Not enough samples in buffer

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert numpy arrays to PyTorch tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(-1).to(self.device) # (batch_size, 1)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Get Q-values for current states from policy_net
        # We need Q(s,a) for actions actually taken
        current_q_values = self.policy_net(states_t).gather(1, actions_t) # Gathers Q-values for chosen actions

        # Get max Q-values for next states from target_net (Double DQN: use policy_net to select best action, target_net to evaluate it)
        # For standard DQN, use target_net for both selection and evaluation of next state Qs.
        with torch.no_grad():
            next_q_values_target = self.target_net(next_states_t).max(1)[0]
            # Compute target Q-values: reward + gamma * max_a' Q_target(s', a')
            # If done, target Q is just the reward
            target_q_values = rewards_t + (self.gamma * next_q_values_target * (1 - dones_t))

        # Compute loss (e.g., Huber loss or MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1)) # Huber loss
        # loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1)) # MSE loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but often helpful)
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_net_if_needed(self):
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated at step {self.steps_done}")

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
        self.policy_net.eval() # Set to evaluation mode if only for inference
        self.target_net.eval()
        print(f"Model loaded from {path}")

def get_epsilon(steps_done):
    """Calculates epsilon value based on the number of steps done."""
    epsilon = EPS_END + (EPS_START - EPS_END) * \
              math.exp(-1. * steps_done / EPS_DECAY_FRAMES)
    return epsilon
def train_agent():
    print("Starting training...")
    # Environment setup
    # `continuous=False` gives discrete actions by default, but we use our own mapping.
    # It's mainly to ensure the action space is treated as discrete by wrappers if they check.
    # The actual actions sent to env.step() will be our continuous vectors.
    env_train = gym.make(ENV_NAME, continuous=True, render_mode=None) # Use continuous=True to send our own vectors
    env_train.action_space = gym.spaces.Discrete(NUM_ACTIONS) # Lie about action space for compatibility if needed by wrappers
                                                            # This isn't strictly necessary if we handle action mapping directly

    # Apply wrappers
    env_train = GrayScaleObservation(env_train, keep_dim=False) # Output: (H, W)
    env_train = FrameStack(env_train, num_stack=FRAME_STACK_SIZE, lz4_compress=True) # Output: (num_stack, H, W)

    agent = DQNAgent(input_channels=INPUT_CHANNELS, height=IMG_HEIGHT, width=IMG_WIDTH,
                     n_actions=NUM_ACTIONS, device=device, lr=LEARNING_RATE,
                     gamma=GAMMA, target_update_freq=TARGET_UPDATE_FREQ)

    episode_rewards = []
    losses = []
    total_steps_overall = 0

    for i_episode in range(NUM_EPISODES_TRAIN):
        state, info = env_train.reset(seed=SEED + i_episode) # state is LazyFrames (FRAME_STACK_SIZE, H, W)
        current_episode_reward = 0
        episode_loss = 0
        num_optim_steps = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            epsilon = get_epsilon(agent.steps_done)
            action_idx = agent.select_action(state, epsilon)
            actual_action_vec = DISCRETE_ACTIONS_LIST[action_idx]

            next_state, reward, terminated, truncated, info = env_train.step(actual_action_vec)
            done = terminated or truncated

            agent.store_transition(state, action_idx, reward, next_state, done)

            loss_val = agent.optimize_model(BATCH_SIZE)
            if loss_val is not None:
                episode_loss += loss_val
                num_optim_steps +=1

            agent.update_target_net_if_needed() # Updates based on agent.steps_done

            state = next_state
            current_episode_reward += reward
            total_steps_overall +=1

            if done:
                break

        episode_rewards.append(current_episode_reward)
        avg_loss = episode_loss / num_optim_steps if num_optim_steps > 0 else 0
        losses.append(avg_loss)

        print(f"Episode {i_episode+1}/{NUM_EPISODES_TRAIN} | Steps: {t+1} | Total Steps: {agent.steps_done} | "
              f"Reward: {current_episode_reward:.2f} | Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.4f}")

        # Save model periodically
        if (i_episode + 1) % 50 == 0 or i_episode == NUM_EPISODES_TRAIN -1 :
            agent.save_model(MODEL_PATH)

    env_train.close()
    print("Training finished.")

    # Plotting rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # Calculate and plot moving average of rewards
    moving_avg_rewards = np.convolve(episode_rewards, np.ones(20)/20, mode='valid') # 20-episode moving average
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg_rewards)
    plt.title('Moving Average of Episode Rewards (20 episodes)')
    plt.xlabel('Episode (start of window)')
    plt.ylabel('Average Reward')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_rewards_pytorch.png"))
    plt.show()


# --- 5. Evaluate Model and Record Video ---
def evaluate_and_record_agent(num_eval_episodes=5):
    print("\nStarting evaluation and video recording...")

    # Environment setup for evaluation
    # The `render_mode` must be "rgb_array" for RecordVideo.
    # We also use `continuous=True` to send our custom action vectors.
    eval_env = gym.make(ENV_NAME, continuous=True, render_mode="rgb_array")

    # Apply wrappers (same as training, except for RecordVideo)
    eval_env = GrayScaleObservation(eval_env, keep_dim=False)
    eval_env = FrameStack(eval_env, num_stack=FRAME_STACK_SIZE, lz4_compress=True)

    # Wrap with RecordVideo. Ensure VIDEO_DIR exists.
    # `episode_trigger` defines when to record. lambda x: True records all episodes.
    # `video_length=0` means record the whole episode.
    eval_env = RecordVideo(eval_env, VIDEO_DIR,
                           episode_trigger=lambda episode_id: True, # Record all eval episodes
                           name_prefix=f"dqn-carracing-eval",
                           video_length=MAX_STEPS_PER_EPISODE) # Max length of video

    agent = DQNAgent(input_channels=INPUT_CHANNELS, height=IMG_HEIGHT, width=IMG_WIDTH,
                     n_actions=NUM_ACTIONS, device=device)
    try:
        agent.load_model(MODEL_PATH)
        agent.policy_net.eval() # Set to evaluation mode
        print("Loaded trained model for evaluation.")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}. Cannot evaluate.")
        eval_env.close()
        return

    total_eval_rewards = []

    for i_episode in range(num_eval_episodes):
        state, info = eval_env.reset(seed=SEED + 1000 + i_episode) # Use a different seed offset for eval
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated) and step_count < MAX_STEPS_PER_EPISODE :
            # In evaluation, use epsilon = 0 (greedy policy)
            action_idx = agent.select_action(state, epsilon=0.0) # Epsilon = 0 for greedy
            actual_action_vec = DISCRETE_ACTIONS_LIST[action_idx]

            next_state, reward, terminated, truncated, info = eval_env.step(actual_action_vec)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            step_count +=1
            # eval_env.render() # RecordVideo handles rendering for saving

        total_eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {i_episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")

    eval_env.close() # This is important to save the last video properly
    avg_reward = sum(total_eval_rewards) / num_eval_episodes if num_eval_episodes > 0 else 0
    print(f"\nEvaluation finished. Average reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    print(f"Videos saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    # Step 1: Train the agent
    # Set NUM_EPISODES_TRAIN to a higher value (e.g., 1000+) for better performance.
    # For a quick test, you can use a smaller number.
    # To skip training and only evaluate a pre-trained model, comment out train_agent().
    train_agent()

    # Step 2: Evaluate the trained agent and record video
    evaluate_and_record_agent(num_eval_episodes=3)