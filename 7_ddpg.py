import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def evaluate_model(model, env, num_episodes=20000):
    """
    Evaluate trained model by running model a number of episodes and calculate total rewards of each episode.

    :param model: Trained model
    :param env: Training environment
    :param num_episodes: number of episodes to run
    :return: total rewards for each episode
    """
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            # Render the environment during evaluation
            env.render()
        episode_rewards.append(episode_reward)
    return episode_rewards


# Create environment
env = gym.make("Pendulum-v1", render_mode="human")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create model
model = DDPG("MlpPolicy", env, learning_rate=1000, buffer_size=200000, gamma=0.98, train_freq=1, gradient_steps=1,
             action_noise=action_noise, verbose=1)
# Train
model.learn(total_timesteps=20000, log_interval=10)
model.save("ddpg_pendulum")

# Get environment for evaluation
vec_env = model.get_env()

# Load
model = DDPG.load("ddpg_pendulum")
# Evaluate
episode_rewards = evaluate_model(model, vec_env)
# Average return
average_return = np.mean(episode_rewards)

# Plot learning curve
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DDPG Learning Curve')
plt.show()

print(f'Average Return: {average_return}')