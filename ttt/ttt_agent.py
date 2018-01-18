from random import choice
import abc
import numpy as np
import os
import pickle
from genetic import Genetic, Genome
from ttt_models import PolicyGradientModel, MCTSModel, RandomModel

class TTTAgent():
    def __init__(self):
        self.model = self.buildmodel()

    @abc.abstractmethod
    def buildmodel(self):
        return None

    def next_move(self, board, print_board = True):
        mov = self.model.predict(board)
        board.add_piece(mov)

        if print_board:
            print(board)

        return mov

    def save_dataset(self,filename, data):
        pickle.dump(data, open(filename, 'wb+'))

    def read_dataset(self,filename):
        if os.path.isfile(filename):
            return pickle.load(open(filename, 'rb'))
        return None

    def update_model(self, player, winner, board_memory, move_memory, learning_rate):
        pass

class TTTGeneticAgent(TTTAgent):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__()
        
    def buildmodel(self):
        population_size = 10
        if self.read_dataset(self.model_name) is None:
            genomes = [Genome() for _ in range(population_size)]
            current_genome_idx = -1
        else:
            genomes, current_genome_idx = self.read_dataset(self.model_name)
        return Genetic(genomes, current_genome_idx, population_size)
    
    def update_model(self, player, winner, board_memory, move_memory, learning_rate):
        if player == winner:
            score = 10
        elif winner == 0:
            score = 2
        else:
            score = -5

        self.model.update(score)
        self.save_dataset(self.model_name, (self.model.genomes, self.model.current_genome))

    def get_current_model(self):
        return self.model.genomes[self.model.current_genome_idx]

class TTTPolicyGradientAgent(TTTAgent):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__()


    def buildmodel(self):
        H = 1000
        weights = self.read_dataset(self.model_name)

        if weights is None:
            weights = {
                      "W1" : np.random.rand(H, 9) * 2 - 1,
                      "W2" : np.random.rand(9, H) * 2 - 1
                      }
        
        return PolicyGradientModel(weights, self.model_name)

    def update_model(self, player, winner, board_memory, move_memory, learning_rate):
        self.model.update_weights(player, winner, board_memory, move_memory, learning_rate)
        self.save_dataset(self.model.name, self.model.weights)

    def get_current_model(self):
        return self.model.weights

class TTTMCTSAgent(TTTAgent):
    def buildmodel(self):
        return MCTSModel()
        
        
class TTTRandomAgent(TTTAgent):
    def buildmodel(self):
        return RandomModel()

