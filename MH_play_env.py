"""
Instead of visualizing the model for the final run, I instead replay the best trajectory of the model,
which I believe is more informative to how well the model performs

This also takes into account the extra levels that were added to the game (I stole levels from Michael)
"""

from MH_env import ShooterEnv
import json

def load_trajectory(file_path="models/best_trajectory.json"):
    """ Load the trajectory from a JSON file """

    with open(file_path, "r") as f:
        trajectory = json.load(f)
    return trajectory

def replay_trajectory(env, trajectory):
    """ Replay the trajectory in the environment """

    state, info = env.reset()
    env.render_mode = 'human'  # Enable rendering for visualization

    # Main game loop
    for step in trajectory:
        action = step["action"]
        reward = step["reward"]
        done = step["done"]

        # Perform the action in the environment
        next_state, _, terminated, truncated, _ = env.step(action)

        # Render the environment
        env.render()

        # Check if the level is complete
        if env.game.level_complete:
            if not env.game.load_next_level():
                print("All levels completed.")
                break  # Exit if there are no more levels
            else:
                # Reset the environment for the next level
                state, info = env.reset()
                continue

        # Check if the episode ends
        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":

    # Load the saved trajectory
    trajectory = load_trajectory("models/best_trajectory.json")

    # Load the environment
    env = ShooterEnv(render_mode='human')  # Enable rendering

    # Replay the trajectory
    replay_trajectory(env, trajectory)