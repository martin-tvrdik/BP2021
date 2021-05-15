import gym
import neat
import pickle
import time

from RL.wrappers import PytorchRAMWrapper

ENV_NAME = "MsPacman-ram-v4"
CFG_FILE = "./config"
MODELS_FILE = "./models/"
MODEL_NAME = "MsPacman-ram-v4_best_genome.pkl"


env = gym.make(ENV_NAME)
env = PytorchRAMWrapper(env)

observation = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, CFG_FILE)

with open(MODELS_FILE + MODEL_NAME, "rb") as f:
    best_bot = pickle.load(f)

net = neat.nn.feed_forward.FeedForwardNetwork.create(best_bot, config)

state = env.reset()

done = False
total_reward = 0

while not done:
    state = state.flatten() / 255.
    output = net.activate(state)
    action = output.index(max(output))
    observation, reward, done, info, additional_done = env.step(action)
    state = observation
    total_reward += reward
    time.sleep(0.05)
    env.render()

print(total_reward)
