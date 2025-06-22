import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class PolicyGradientAgent:
    def __init__(self, state_size, action_size, learning_rate=2e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.states = []
        self.actions = []
        self.rewards = []
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        
        # Store state and action for training
        self.states.append(state)
        self.actions.append(action)
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def learn(self):
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in self.rewards[::-1]:  # Reverse the rewards
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Convert states and actions to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        
        # Calculate loss
        probs = self.policy(states)
        m = Categorical(probs)
        loss = -m.log_prob(actions) * discounted_rewards
        loss = loss.sum()
        
        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return loss.item()
    
    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
    
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))

def train_pg(env_name='CartPole-v1', num_episodes=500, max_steps=200, render_every=100):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = PolicyGradientAgent(state_size, action_size)
    scores = []
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward)
            score += reward
            state = next_state
            
            if done:
                break
        
        # Learn after episode completion
        loss = agent.learn()
        scores.append(score)
        
        # Print episode statistics
        if episode % 10 == 0:
            avg_score = np.mean(scores[-20:])
            print(f"Episode {episode}/{num_episodes}, Avg Score: {avg_score:.2f}")
    
    return agent, scores