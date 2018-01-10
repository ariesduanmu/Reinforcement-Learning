import numpy as np
import random
from tictactoe_game import TicTacToeBoard
from ttt_agent import TTTPolicyGradientAgent, TTTRandomAgent

model_name1 = "model1"
model_name2 = "model2"

def play_game(p1, p2):
    board = TicTacToeBoard()
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
    for i in range(episodes):
        p1 = TTTPolicyGradientAgent(1, model_name1)
        p2 = TTTPolicyGradientAgent(-1, model_name2)

        winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory = play_game(p1, p2)
        p1.update_model(1, winner, p1_board_memory, p1_move_memory, learning_rate)
        p2.update_model(-1, winner, p2_board_memory, p2_move_memory, learning_rate)
        

def test():
    results = [0, 0, 0]

    for _ in range(1000):
        r = random.choice([1, -1])
        s = -1 * r
        model_name = model_name1 if s == 1 else model_name2

        random_player = TTTRandomAgent(r)
        smart_player = TTTPolicyGradientAgent(s, model_name)

        if r == 1:
            winner, _, _, _, _ = play_game(random_player, smart_player)
        else:
            winner, _, _, _, _ = play_game(smart_player, random_player)

        results[abs(winner - s)] += 1
        

    print("win rate: {}\n win - {r[0]} lose - {r[1]} draw - {r[2]}".format(results[0] / sum(results), r=results))

if __name__ == "__main__":
    for _ in range(10):
        #train(10000, 0.2)
        test()




