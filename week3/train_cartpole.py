"""
REINFORCE Training Script for CartPole

This script can be run locally on your laptop to train a policy gradient agent.

Usage:
    python train_cartpole.py --episodes 1000 --lr 0.01 --gamma 0.99
    python train_cartpole.py --visualize --model trained_policy.pth
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from pathlib import Path


class PolicyNetwork(nn.Module):
    """Neural network policy for CartPole."""
    
    def __init__(self, state_dim=4, hidden_dim=32, action_dim=2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """Forward pass: state → action logits."""
        return self.network(state)
    
    def select_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state: numpy array
            deterministic: if True, select argmax (for evaluation)
        
        Returns:
            action, log_prob
        """
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs).item()
            log_prob = torch.log(probs[action])
        else:
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action])
        
        return action, log_prob


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.FloatTensor(returns)


def train_reinforce(env, policy, optimizer, n_episodes=1000, gamma=0.99, 
                     save_path=None, plot=True):
    """Train policy using REINFORCE algorithm."""
    episode_rewards = []
    best_avg_reward = -float('inf')
    
    print(f"Training for {n_episodes} episodes...")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Gamma: {gamma}\n")
    
    for episode in range(n_episodes):
        log_probs = []
        rewards = []
        
        state, _ = env.reset()
        done = False
        
        while not done:
            action, log_prob = policy.select_action(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        
        # Normalize returns (variance reduction)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        loss = sum(-lp * G for lp, G in zip(log_probs, returns))
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track performance
        episode_rewards.append(sum(rewards))
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1:4d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Last Episode: {episode_rewards[-1]:3.0f}")
            
            # Save best model
            if save_path and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(policy.state_dict(), save_path)
                print(f"  → Saved new best model (avg: {avg_reward:.2f})")
    
    print(f"\nTraining complete!")
    print(f"Final 100-episode average: {np.mean(episode_rewards[-100:]):.2f}")
    
    if plot:
        plot_training_curve(episode_rewards)
    
    return episode_rewards, policy


def plot_training_curve(rewards, save_path='training_curve.png'):
    """Plot and save training curve."""
    plt.figure(figsize=(12, 5))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode reward')
    
    # Moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                 color='red', linewidth=2, label=f'{window}-episode MA')
    
    plt.axhline(y=475, color='g', linestyle='--', alpha=0.7, label='Success threshold')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('REINFORCE Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution
    plt.subplot(1, 2, 2)
    early = rewards[:min(200, len(rewards)//2)]
    late = rewards[max(len(rewards)//2, len(rewards)-200):]
    
    plt.hist(early, bins=20, alpha=0.5, label='Early episodes')
    plt.hist(late, bins=20, alpha=0.5, label='Late episodes')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved training curve to {save_path}")
    plt.show()


def visualize_policy(policy, n_episodes=5):
    """Visualize trained policy."""
    env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = policy.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train REINFORCE on CartPole')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Hidden layer size (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save-model', type=str, default='trained_policy.pth',
                        help='Path to save trained model (default: trained_policy.pth)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize trained policy (requires --model)')
    parser.add_argument('--model', type=str, default='trained_policy.pth',
                        help='Path to model for visualization')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment and policy
    env = gym.make('CartPole-v1')
    policy = PolicyNetwork(state_dim=4, hidden_dim=args.hidden_dim, action_dim=2)
    
    if args.visualize:
        # Load and visualize
        if Path(args.model).exists():
            policy.load_state_dict(torch.load(args.model))
            print(f"Loaded model from {args.model}")
            visualize_policy(policy)
        else:
            print(f"Error: Model file {args.model} not found!")
            return
    else:
        # Train
        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        
        train_reinforce(
            env, policy, optimizer,
            n_episodes=args.episodes,
            gamma=args.gamma,
            save_path=args.save_model,
            plot=not args.no_plot
        )
        
        print(f"\nModel saved to {args.save_model}")
        print(f"To visualize: python {__file__} --visualize --model {args.save_model}")


if __name__ == "__main__":
    main()
