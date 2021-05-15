import gym
from RL.wrappers import PytorchRAMWrapper
import torch

ENV_NAME = 'MsPacman-ram-v4'
RANDOM_ATTEMPTS = 100

env = gym.make(ENV_NAME)
env = PytorchRAMWrapper(env, add_done=True)

image = env.reset()
image = torch.tensor(image)

total_reward = 0
total_score = 0
for i in range(RANDOM_ATTEMPTS):

    done = False
    while not done:
        action = env.action_space.sample()

        image, reward, done, info, _ = env.step(action)
        image = torch.tensor(image)
        # env.render()
        total_reward += reward
        if done:
            total_score += info["score"]
            image = env.reset()
            image = torch.tensor(image)

print("avg reward: " + str(total_reward/RANDOM_ATTEMPTS) + ", avg score: " + str(total_score/RANDOM_ATTEMPTS))
#  Alien-ram-v4:            avg reward: 15.84, avg score: 159.7
#  Boxing-ram-v4:           avg reward: 0.63, avg score: 0.39
#  MsPacman-ram-v4:         avg reward: 22.49, avg score: 225.3
#  Pong-ram-v4:             avg reward: -20.12, avg score: -20.12
