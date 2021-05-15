import gym
import neat
import numpy as np
import pickle
import GA.visualize as visualize

from RL.wrappers import PytorchRAMWrapper

CHECKPOINTS_FILE = "checkpoints/"
CFG_FILE = "./config"
MODELS_FILE = "models/"
PLOTS_FILE = "plots/"
DATA_FILE = "evolution_data/"

NUM_GENERATIONS = 200

ENV_NAME = "MsPacman-ram-v4"

env = gym.make(ENV_NAME)
env = PytorchRAMWrapper(env)

print(env.action_space)

gen_best_reward = []
gen_average_reward = []


def eval_genomes(genomes, cfg):
    min_best_gen_reward = -10000

    for genome_id, genome in genomes:
        state = env.reset()
        high_score = 0
        frame = 0
        net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, cfg)

        while True:
            frame += 1

            state = state.flatten() / 255.
            # print(state)
            output = net.activate(state)
            action = output.index(max(output))
            # print(action)
            observation, reward, done, info, additional_done = env.step(action)
            state = observation

            # env.render()
            high_score += reward
            if done or additional_done:
                break
        fitness = high_score

        if fitness > min_best_gen_reward:
            min_best_gen_reward = fitness

        genome.fitness = fitness

    gen_best_reward.append(min_best_gen_reward)
    print(gen_best_reward)
    mean_gen_best_reward_last_10 = round(np.mean(gen_best_reward[-11:-1]), 1)
    with open(DATA_FILE + ENV_NAME + ".csv", 'a') as file:
        if len(gen_best_reward) > 10:
            file.write(str(min_best_gen_reward) + ","
                       + str(mean_gen_best_reward_last_10))
            file.write('\n')
        else:
            file.write(str(min_best_gen_reward))
            file.write('\n')


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, CFG_FILE)

p = neat.Population(config)
# p = neat.Checkpointer.restore_checkpoint(CHECKPOINTS_FILE + "MsPacman-ram-v0_297") # resume evolution from checkpoint
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10, filename_prefix=CHECKPOINTS_FILE + ENV_NAME + "_"))

best_genome = p.run(eval_genomes, NUM_GENERATIONS)

pickle.dump(best_genome, open(MODELS_FILE + ENV_NAME + '_best_genome.pkl', 'wb'))

visualize.draw_net(config, best_genome, True, filename=PLOTS_FILE + ENV_NAME + "_diagraph")
visualize.plot_stats(stats, ylog=False, view=True, filename=PLOTS_FILE + ENV_NAME + "_" + "avg_fitness")
visualize.plot_species(stats, view=True, filename=PLOTS_FILE + ENV_NAME + "_" + "speciation")
