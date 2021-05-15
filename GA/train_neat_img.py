import neat
import cv2
import pickle
#  import visualize
import GA.visualize as visualize
import gym
import numpy as np

CHECKPOINTS_FILE = "checkpoints/"
CFG_FILE = "./config_img"
MODELS_FILE = "models/"
PLOTS_FILE = "plots/"

NUM_GENERATIONS = 50

ENV_NAME = "MsPacmanNoFrameskip-v4"

env = gym.make(ENV_NAME)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        state = env.reset()
        high_score = 0
        frame = 0
        # create f-f network from genome
        net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)

        while True:
            frame += 1

            img = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ob = img
            ob = cv2.resize(ob, (42, 42))
            flatten_img = np.ndarray.flatten(ob)

            output = net.activate(flatten_img)
            action = np.argmax(output)
            # print(action)
            observation, reward, done, info = env.step(action)
            state = observation

            #env.render()
            high_score += reward
            if done:
                break
        fitness = high_score

        genome.fitness = fitness


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, CFG_FILE)

p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint(CHECKPOINTS_FILE + "MsPacmanNoFrameskip-v4_49") # to resume evolution from checkpoint
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=1200, filename_prefix=CHECKPOINTS_FILE + ENV_NAME + "_"))

best_genome = p.run(eval_genomes, NUM_GENERATIONS)

pickle.dump(best_genome, open(MODELS_FILE + ENV_NAME + '_best_genome.pkl', 'wb'))

visualize.draw_net(config, best_genome, True, filename=PLOTS_FILE + ENV_NAME + "_diagraph")
visualize.plot_stats(stats, ylog=False, view=True, filename=PLOTS_FILE + ENV_NAME + "_" + "avg_fitness")
visualize.plot_species(stats, view=True, filename=PLOTS_FILE + ENV_NAME + "_" + "speciation")
