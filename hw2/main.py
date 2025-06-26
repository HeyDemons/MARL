import argparse
import gymnasium as gym
import torch
import numpy as np
from dqn import train_dqn, DQNAgent
from policy_gradient import train_pg, PolicyGradientAgent
from utils import plot_training_results, evaluate_agent, record_video
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate agents on CartPole")
    parser.add_argument("--algorithm", type=str, default="pg", choices=["dqn", "pg", "both"], 
                      help="Algorithm to train (dqn, pg, or both)")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--eval", action="store_true", help="Evaluate the trained agent")
    parser.add_argument("--record", action="store_true", help="Record a video of the agent")
    parser.add_argument("--load", action="store_true", help="Load trained model instead of training")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")
    
    args = parser.parse_args()
    
    env_name = 'CartPole-v1'
    
    # Initialize variables
    dqn_agent = None
    pg_agent = None
    dqn_scores = None
    pg_scores = None
    
    # Train or load DQN agent
    if args.algorithm in ["dqn", "both"]:
        if args.load:
            env = gym.make(env_name)
            dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
            try:
                dqn_agent.load("dqn_cartpole.pth")
                print("DQN model loaded successfully")
            except:
                print("Failed to load DQN model, please train first.")
        else:
            print("Training DQN agent...")
            dqn_agent, dqn_scores = train_dqn(env_name, num_episodes=args.episodes)
            dqn_agent.save("dqn_cartpole.pth")
            print("DQN training complete")
    
    # Train or load Policy Gradient agent
    if args.algorithm in ["pg", "both"]:
        if args.load:
            env = gym.make(env_name)
            pg_agent = PolicyGradientAgent(env.observation_space.shape[0], env.action_space.n)
            try:
                pg_agent.load("pg_cartpole.pth")
                print("Policy Gradient model loaded successfully")
            except:
                print("Failed to load Policy Gradient model, please train first.")
        else:
            print("Training Policy Gradient agent...")
            pg_agent, pg_scores = train_pg(env_name, num_episodes=args.episodes)
            pg_agent.save("pg_cartpole.pth")
            print("Policy Gradient training complete")
    
    # Plot training results
    if dqn_scores is not None or pg_scores is not None:
        plot_training_results(dqn_scores, pg_scores, window=20)
    
    # Evaluate agents
    if args.eval:
        if dqn_agent:
            print("Evaluating DQN agent...")
            dqn_score = evaluate_agent(dqn_agent, env_name, render=args.render, max_steps=200)
            
        if pg_agent:
            print("Evaluating Policy Gradient agent...")
            pg_score = evaluate_agent(pg_agent, env_name, render=args.render, max_steps=200)
            
        if dqn_agent and pg_agent:
            print(f"Comparison: DQN avg score = {dqn_score:.2f}, PG avg score = {pg_score:.2f}")
    
    # Record video
    if args.record:
        if dqn_agent:
            print("Recording DQN agent performance...")
            record_video(dqn_agent, env_name, video_length=500, algorithm_name="dqn")
            
        if pg_agent:
            print("Recording Policy Gradient agent performance...")
            record_video(pg_agent, env_name, video_length=500, algorithm_name="policy_gradient")

if __name__ == "__main__":
    main()