"""
Within the 'models' folder, you will see the model files and: 
the checkpoint/current model in case of early stopping of the program,
and also the 'trajectory' files which are used for replay/debugging purposes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from MH_env import ShooterEnv
import pygame
import os
import json

# Ensure the 'models' folder exists
os.makedirs("models", exist_ok=True)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05
target_update_freq = 10
batch_size = 64
max_episodes = 1000 # originally 500

# Tallman Code
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

# Tallman Code
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

def train_episode(env, online_net, stable_net, replay, optimizer):
    """ Train the agent for one episode """

    # Starting variables
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0

    # Track the trajectory for this episode
    # Trajectory is the list of steps taken in the episode - this is used for replay purposes
    trajectory = []

    # Track the furthest horizontal distance - this and reward are used in tandem for training
    furthest_distance = env.game.player.rect.centerx

    while not done:

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = online_net(state).argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Save the current step to the trajectory
        trajectory.append({
            "state": state.numpy().tolist(),
            "action": action,
            "reward": reward,
            "next_state": next_state.tolist(),
            "done": terminated or truncated
        })

        # Update the furthest horizontal distance
        furthest_distance = max(furthest_distance, env.game.player.rect.centerx)

        # Check if the level is complete
        if env.game.level_complete:
            if not env.game.load_next_level():
                print("All levels completed. Restarting from level 1.")
                env.game.level = 1  # Restart from the first level - never used haha
                env.reset()
            else:
                # Reset the environment for the next level
                state, info = env.reset()
                state = torch.tensor(state, dtype=torch.float32)
                continue

        # Check if the episode should end
        done = terminated or truncated

        # Render the game if in human mode
        if env.render_mode == 'human':
            env.render()

        # Store the transition in the replay buffer
        replay.push((state.numpy(), action, reward, next_state, done))
        state = torch.tensor(next_state, dtype=torch.float32)

        # Perform a training step if the replay buffer has enough samples - AI help
        if len(replay) >= batch_size:
            states, actions, rewards, next_states, dones = replay.sample(batch_size)
            q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q_values = stable_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_reward, trajectory, furthest_distance

def train_dqn():
    """ Main training loop """

    global epsilon
    env = ShooterEnv(render_mode=None)  # Default render_mode is None
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online_net = DQN(obs_size, n_actions)
    stable_net = DQN(obs_size, n_actions)
    stable_net.load_state_dict(online_net.state_dict())

    # Initialize optimizer and scheduler - scheduler is used to decay the learning rate, AI gave me this suggestion
    optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Load saved model if it exists - this is used for in case training loop was ended early
    model_path = "models/dqn_model.pth"
    if os.path.exists(model_path):
        online_net.load_state_dict(torch.load(model_path))
        print("Loaded saved model.")

    # Create the best reward
    best_reward = float('-inf')

    # Track the furthest distance achieved
    best_distance = float('-inf')
    best_trajectory = None  # To store the best trajectory

    # Load checkpoint - definitely AI help
    checkpoint_path = "models/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        online_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_reward = checkpoint['best_reward']
        epsilon = checkpoint['epsilon']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint with best reward: {best_reward:.2f}")

    replay = ReplayBuffer()
    replay_buffer_path = "models/replay_buffer.pth"

    # Load replay buffer if it exists - this is used for in case training loop was ended early
    if os.path.exists(replay_buffer_path):
        replay.buffer = torch.load(replay_buffer_path, weights_only=False)
        print("Loaded replay buffer.")

    # Calculate intervals for saving intermediate models
    save_intervals = [max_episodes // 4, max_episodes // 2, 3 * max_episodes // 4]

    for episode in range(1, max_episodes + 1):
        # Reinitialize the environment with rendering enabled every 10th episode
        if episode % 10 == 0:
            env = ShooterEnv(render_mode='human')  # Enable rendering
        else:
            env = ShooterEnv(render_mode=None)  # Disable rendering

        total_reward, trajectory, furthest_distance = train_episode(env, online_net, stable_net, replay, optimizer)

        # Save the first model
        if episode == 1:
            torch.save(online_net.state_dict(), "models/first_dqn_model.pth")
            print("Saved first trained model.")

        # Save intermediate models
        if episode in save_intervals:
            torch.save(online_net.state_dict(), f"models/intermediate_dqn_model_{save_intervals.index(episode) + 1}.pth")
            print(f"Saved intermediate model at episode {episode}.")

        # Save the final model and its trajectory
        if episode == max_episodes:
            torch.save(online_net.state_dict(), "models/final_dqn_model.pth")
            print("Saved final trained model.")

            # Save the final trajectory
            final_trajectory_path = "models/final_trajectory.json"
            final_trajectory = make_json_serializable(trajectory)
            with open(final_trajectory_path, "w") as f:
                json.dump(final_trajectory, f)
            print("Saved final trajectory.")

        # Update the best trajectory and model if the furthest distance is greater
        if furthest_distance > best_distance:
            best_distance = furthest_distance
            best_trajectory = make_json_serializable(trajectory)  # Preprocess the trajectory
            torch.save(online_net.state_dict(), model_path)
            print(f"New best distance: {best_distance}. Updated best trajectory and model.")

            # Save the best trajectory to a file
            trajectory_path = "models/best_trajectory.json"
            with open(trajectory_path, "w") as f:
                json.dump(best_trajectory, f)

        # Update if new best reward has been achieved
        if total_reward > best_reward:
            best_reward = total_reward
            print(f"New best reward: {best_reward:.2f}.")

        # Save the replay buffer
        if episode % target_update_freq == 0:
            stable_net.load_state_dict(online_net.state_dict())

        # Debugging prints for episode count, reward, epsilon, and distance of run
        print(f"Episode {episode:>4} | Reward: {total_reward:>10.2f} | Furthest Distance: {furthest_distance} | Epsilon: {epsilon:.3f}")
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        scheduler.step()

        # Save the full checkpoint
        checkpoint_path = "models/checkpoint.pth"
        torch.save({
            'model_state_dict': online_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_reward': best_reward,
            'best_distance': best_distance,
            'epsilon': epsilon,
            'scheduler_state_dict': scheduler.state_dict()
        }, checkpoint_path)

    env.close()

# Helper function to make the trajectory JSON-serializable - AI helped because before the JSON was unable to be read with how loop works
def make_json_serializable(trajectory):
    """ Convert the trajectory to a JSON-serializable format """

    for step in trajectory:
        step["state"] = [float(x) for x in step["state"]]  # Convert state to a list of floats
        step["next_state"] = [float(x) for x in step["next_state"]]  # Convert next_state to a list of floats
        step["action"] = int(step["action"])  # Convert action to an int
        step["reward"] = float(step["reward"])  # Convert reward to a float
        step["done"] = bool(step["done"])  # Convert done to a boolean
    return trajectory

if __name__ == "__main__":
    train_dqn()