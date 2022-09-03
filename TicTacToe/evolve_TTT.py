import os
import neat
from TTT import TTT

ROOT = os.path.dirname(os.path.realpath(__file__))
T = TTT()

def eval_genomes(genomes, config):
    for id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(
            genome,
            config
        )