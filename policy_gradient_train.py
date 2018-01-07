import numpy as np
import random
from utitly import *
from tictactoe_game import TicTacToeBoard

class TTTAgent():
    def set_player(self, player):
        self.player = player

    def get_player(self):
        return self.player

    def next_move(self, board, print_board = True):
        empty_positions = board.get_empty_pos()
        mov = random.choice(empty_positions)
        board.add_piece(self.player, mov)

        if print_board:
            print(board)

        return mov
class TTTSmartAgent(TTTAgent):
    def __init__(self, model):
        self.model = model

    

    def next_move(self, board, print_board = True):
        board_input = np.asarray(board.get_board()).reshape(9,1)
        _,_,_,out = feed_forward(self.model, board_input)
        try:
            out_norm = [float(x) / float(sum(out)) for x in out]
            mov = int(np.random.choice(9, 1, p=out_norm))
        except ZeroDivisionError:
            mov = -1
        empty_positions = board.get_empty_pos()
        if mov not in empty_positions or mov == -1:
            mov = random.choice(empty_positions)
        board.add_piece(self.player, mov)

        if print_board:
            print(board)
        return mov

def process_training_data(player, winner, board_memory, move_memory):

    X = np.array(board_memory).T
    if winner == player.get_player():
        moves = [[1 if i == m else 0 for i in range(9)] for m in move_memory]
    elif winner == 0:
        moves = [[0.5 if i == m else 0 for i in range(9)] for m in move_memory]
    else:
        moves = [[0 for _ in range(9)] for _ in move_memory]

    y = np.array(moves).T

    return X, y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def feed_forward(model, X):
    h2 = model["W1"] @ X
    a2 = sigmoid(h2)
    h3 = model["W2"] @ h2
    a3 = sigmoid(h3)

    return h2, a2, h3, a3

def update_model(model, player, winner, board_memory, move_memory, learning_rate):

    X, y = process_training_data(player, winner, board_memory, move_memory)

    _, n = y.shape

    h2, a2, h3, a3 = feed_forward(model, X)

    delta3 = a3 - y
    delta2 = model["W2"].T @ delta3 * sigmoid_prime(h2)

    W1_grad = (1 / n) * delta2 @ X.T
    W2_grad = (1 / n) * delta3 @ h2.T

    model["W1"] -= (learning_rate / n) * W1_grad
    model["W2"] -= (learning_rate / n) * W2_grad

    return model


def train_play(p1, p2):
    board = TicTacToeBoard()
    p1.set_player(1)
    p2.set_player(-1)
    p1_board_memory = []
    p2_board_memory = []
    p1_move_memory = []
    p2_move_memory = []

    for _ in range(5):
        p1_board_memory.append(board.get_board().copy())
        p1_move = p1.next_move(board, print_board = False)
        p1_move_memory.append(p1_move)
        if board.is_over()[0]:
            break

        p2_board_memory.append(board.get_board().copy())
        p2_move = p2.next_move(board, print_board = False)
        p2_move_memory.append(p2_move)
        if board.is_over()[0]:
            break
    winner = board.is_over()[1]
    return winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory




def train(episodes, learning_rate):
    
    model1 = read_dataset("model1")
    model2 = read_dataset("model2")

    H = 1000

    if model1 is None:
        model1 = {
                  "W1" : np.random.rand(H, 9) * 2 - 1,
                  "W2" : np.random.rand(9, H) * 2 - 1
                  }

    if model2 is None:
        model2 = {
                  "W1" : np.random.rand(H, 9) * 2 - 1,
                  "W2" : np.random.rand(9, H) * 2 - 1
                  }

    
    for i in range(episodes):
        p1 = TTTSmartAgent(model1)
        p2 = TTTSmartAgent(model2)

        winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory = train_play(p1, p2)
        model1 = update_model(model1, p1, winner, p1_board_memory, p1_move_memory, learning_rate)
        model2 = update_model(model2, p2, winner, p2_board_memory, p2_move_memory, learning_rate)

    save_dataset("model1", model1)
    save_dataset("model2", model2)

def test():
    model1 = read_dataset("model1")
    model2 = read_dataset("model2")

    win = 0
    lose = 0
    draw = 0

    for _ in range(1000):
        r = random.choice([1, -1])
        s = -1 * r

        model = model1 if s == 1 else model2

        random_player = TTTAgent()
        random_player.set_player(r)
        smart_player = TTTSmartAgent(model)
        smart_player.set_player(s)

        board = TicTacToeBoard()
        for _ in range(5):
            if r == 1:
                random_player.next_move(board, print_board = False)
                if board.is_over()[0]:
                    break

                smart_player.next_move(board, print_board = False)
                if board.is_over()[0]:
                    break
            else:
                smart_player.next_move(board, print_board = False)
                if board.is_over()[0]:
                    break

                random_player.next_move(board, print_board = False)
                if board.is_over()[0]:
                    break
        winner = board.is_over()[1]
        if winner == s:
            win += 1
        elif winner == 0:
            draw += 1
        else:
            lose += 1

    print("win rate: {}\n win - {} lose - {} draw - {}".format(win / (win + draw + lose), win, lose, draw))

if __name__ == "__main__":
    for _ in range(10):
        train(10000, 0.2)
        test()




