import random
import numpy as np
import pickle
from collections import defaultdict, deque
from policy_value_net import PolicyValueNet
from macts import MCTSPlayer as MCTS_Pure

class TrainPipeline():
    def __init__(self):
        pass

    def get_equi_data(self, play_data):
        pass

    def collect_selfplay_data(self, n_games=1):
        pass

    def policy_update(self):
        pass

    def policy_evaluate(self, n_games=10):
        pass

    def run(self):
        pass
if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    TrainPipeline.run()
