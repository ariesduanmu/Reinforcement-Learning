import numpy as np
from mcts import MCTS
from random import choice



class PolicyGradientModel():
    def __init__(self, weights, name):
        self.weights = weights
        self.name = name

    def predict(self, board):
        board_input = np.asarray(board.get_board()).reshape(9,1)
        _,_,_,out = self.feed_forward(board_input)
        try:
            out_norm = [float(x) / float(sum(out)) for x in out]
            mov = int(np.random.choice(9, 1, p=out_norm))
        except ZeroDivisionError:
            mov = -1
        empty_positions = board.get_empty_pos()
        if mov not in empty_positions or mov == -1:
            mov = choice(empty_positions)
        return mov

    def feed_forward(self, X):
        h2 = self.weights["W1"] @ X
        a2 = self.sigmoid(h2)
        h3 = self.weights["W2"] @ h2
        a3 = self.sigmoid(h3)

        return h2, a2, h3, a3

    def update_weights(self, player, winner, board_memory, move_memory, learning_rate):

        X, y = self.process_training_data(player, winner, board_memory, move_memory)

        _, n = y.shape

        h2, a2, h3, a3 = self.feed_forward(X)

        delta3 = a3 - y
        delta2 = self.weights["W2"].T @ delta3 * self.sigmoid_prime(h2)

        W1_grad = (1 / n) * delta2 @ X.T
        W2_grad = (1 / n) * delta3 @ h2.T

        self.weights["W1"] -= (learning_rate / n) * W1_grad
        self.weights["W2"] -= (learning_rate / n) * W2_grad

    def process_training_data(self, player, winner, board_memory, move_memory):

        X = np.array(board_memory).T
        if winner == player:
            moves = [[1 if i == m else 0 for i in range(9)] for m in move_memory]
        elif winner == 0:
            moves = [[0.5 if i == m else 0 for i in range(9)] for m in move_memory]
        else:
            moves = [[0 for _ in range(9)] for _ in move_memory]

        y = np.array(moves).T

        return X, y

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

class MCTSModel():
    def __init__(self, c = 5, max_move = 2000):
        self.mcts = MCTS(self.policy, c, max_move)

    def predict(self, board):
        empty_positions = board.get_empty_pos()
        if len(empty_positions) > 0:
            move = self.mcts.predict(board)
            self.mcts.update_last_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def policy(self, board):
        action_probs = np.ones(len(board.get_empty_pos()))/len(board.get_empty_pos())
        return zip(board.get_empty_pos(), action_probs), 0

        
class RandomModel():
    def predict(self, board):
        return choice(board.get_empty_pos())



