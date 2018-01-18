import numpy as np
from mcts_utc import MCTS
from random import choice
from nenural_network_numpy import *

class PolicyGradientModel():
    def __init__(self, weights, name):
        self.weights = weights
        self.name = name

    def predict(self, board):
        board_input = np.asarray(board.get_board()).reshape(9,1)
        _,_,_,out = feed_forward(self.weights, board_input)
        try:
            out_norm = [float(x) / float(sum(out)) for x in out]
            mov = int(np.random.choice(9, 1, p=out_norm))
        except ZeroDivisionError:
            mov = -1
        empty_positions = board.get_empty_pos()
        if mov not in empty_positions or mov == -1:
            mov = choice(empty_positions)
        return mov

    def update(self, player, winner, board_memory, move_memory, learning_rate):
        self.weights = update_weights(self.weights, player, winner, board_memory, move_memory, learning_rate)

class MCTSRandomModel():
    def __init__(self, c = 5, seconds = 30):
        self.mcts = MCTS(self.action_policy, self.default_choice, c, seconds)

    def predict(self, board):
        empty_positions = board.get_empty_pos()
        if len(empty_positions) > 0:
            move = self.mcts.search(board)
            self.mcts.update_last_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def action_policy(self, board):
        action_probs = np.ones(len(board.get_empty_pos()))/len(board.get_empty_pos())
        return zip(board.get_empty_pos(), action_probs)

    def default_choice(self, board):
        empty_pos = board.get_empty_pos()
        return choice(empty_pos)

class MCTSNenuralModel(MCTSRandomModel):
    def __init__(self, weights, name, c = 5, seconds = 30):
        self.weights = weights
        self.name = name
        super().__init__(c, seconds)

    def action_policy(self, board):
        board_input = np.asarray(board.get_board()).reshape(9,1)
        _,_,_,out = feed_forward(self.weights, board_input)
        out_norm = [float(x) / float(sum(out)) for x in out]
        action_probs = [out_norm[pos] for pos in board.get_empty_pos()]

        return zip(board.get_empty_pos(), action_probs)

    def default_choice(self, board):
        empty_pos = board.get_empty_pos()
        return choice(empty_pos)

    def update(self, player, winner, board_memory, move_memory, learning_rate):
        self.weights = update_weights(self.weights, player, winner, board_memory, move_memory, learning_rate)

        
class RandomModel():
    def predict(self, board):
        return choice(board.get_empty_pos())



