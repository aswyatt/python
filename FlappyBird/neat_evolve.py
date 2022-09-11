from flappy_bird import *

class FlappyNeat:

    def __init__(self, config_file)
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )
        self.stats = neat.StatisticsReporter()
        self.pop = neat.Population(self.config)
        self.pop.add_reporter(neat.StdOutReporter(True))
        self.pop.add_reporterdd_reporter(self.stats)

    def run():
        self.winner = self.pop.run(self.evolve,50)

    def evolve(self):
        pass

if __name__=="__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "feedforward.cfg")
    flappy = FlappyNeat(config_file)
    flappy.run()