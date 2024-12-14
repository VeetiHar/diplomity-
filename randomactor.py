import gym
import numpy as np
import time
# Environment name
environment_name = "SpaceInvadersNoFrameskip-v4"

# Create the environment
env = gym.make(environment_name, render_mode="rgb_array")

# Number of episodes to evaluate
num_episodes = 1000

# Function to evaluate the random action policy
def evaluate_random_policy(env, num_episodes):
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            env.render()
            action = env.action_space.sample()  # Take a random action
            state, reward, done, truncated , info = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    return total_rewards

# Evaluate the random action policy
rewards = evaluate_random_policy(env, num_episodes)

# Print summary statistics
print(f"Average Reward over {num_episodes} episodes: {np.mean(rewards)}")
print(f"Standard Deviation of Reward over {num_episodes} episodes: {np.std(rewards)}")
print(f"Maximum Reward over {num_episodes} episodes: {np.max(rewards)}")
print(f"Minimum Reward over {num_episodes} episodes: {np.min(rewards)}")