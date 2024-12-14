import gym 
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import time
import warnings
import numpy as np
import torch as th
import torch.nn as nn
from fastonn import SelfONN2d
from torch.optim import RMSprop
import math
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import math

warnings.simplefilter("ignore", category=DeprecationWarning)
def main(res):
    class CustomCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
            super().__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(

                SelfONN2d(n_input_channels, 32, 8, q=3, stride=4, padding=0),
                nn.ReLU(),
                SelfONN2d(32, 64, 4, q=3, stride=2, padding=0),
                nn.ReLU(),
                SelfONN2d(64, 64, 3, q=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            with th.no_grad():
                n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, features_dim),
                nn.ReLU()
            )
        
        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))

    # Create and wrap the environment
    environment_name = 'BreakoutNoFrameskip-v4'
    env = make_atari_env(environment_name, n_envs=16)
    env = VecFrameStack(env, n_stack=4)


    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        optimizer_class=RMSprop,
        optimizer_kwargs=dict(alpha=0.95,eps=1e-5)
    )
    initial_learning_rate = 1e-4
    model = A2C(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=initial_learning_rate,
        ent_coef=0.01,
        vf_coef=0.25,
    )
    policy = model.policy

    # Count the number of parameters
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total number of parameters in the A2C model: {total_params}")


    iterations = 100
    total_timesteps = 100_000
    mean_rewards = np.zeros(iterations)
    std_rewards = np.zeros(iterations)
    training_times = np.zeros(iterations)
    for i in range(iterations):
        start_time = time.time()
        print(model.learning_rate)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        end_time = time.time()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
        mean_rewards[i] = mean_reward
        std_rewards[i] = std_reward
        training_times[i] = end_time-start_time
        print(i+1)
        print('mean reward: ', mean_reward)
        print('std reward', std_reward)
        print('time', end_time-start_time)
    game = 'breakout'
    model_name = 'A2C_10M'
    q = '3'
    result = '_'+str(res)
    directorya = 'arrays'
    directorym = 'models'

    # If the directory does not exist, create it
    if not os.path.exists(os.path.join(game, model_name, q, directorya)):
        os.makedirs(os.path.join(game, model_name, q, directorya))

    # If the directory does not exist, create it
    if not os.path.exists(os.path.join(game, model_name, q, directorym)):
        os.makedirs(os.path.join(game, model_name, q, directorym))

    training_times_path = os.path.join(game,model_name,q,directorya, 'training_times_'+str(total_timesteps*iterations)+result)
    mean_rewards_path = os.path.join(game,model_name,q,directorya, 'mean_rewards_'+str(total_timesteps*iterations)+result)
    std_rewards_path = os.path.join(game,model_name,q,directorya, 'std_rewards_'+str(total_timesteps*iterations)+result)
    models_path = os.path.join(game,model_name,q,directorym,result)

    np.save(training_times_path, training_times)
    np.save(mean_rewards_path, mean_rewards)
    np.save(std_rewards_path, std_rewards)
    model.save(models_path)
for i in range(1):
    main(i+4)