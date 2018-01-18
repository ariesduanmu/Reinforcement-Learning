import numpy as np
import datetime
import copy
from random import choice

class TTTBoard:
    def __init__(self):
        self.board = [0 for _ in range(9)]

    def restart(self):
        self.board = [0 for _ in range(9)]

    def get_board(self):
        return self.board

    def get_empty_pos(self):
        return [index for index, value in enumerate(self.get_board()) if value == 0]

    @property
    def current_player(self):
        return 1 if sum(self.board) == 0 else -1

    def add_piece(self, move):
        if move in self.get_empty_pos():

            self.board[move] = self.current_player

    def is_over(self):
        board = self.get_board()

        if board[0] != 0 and len(set(board[:3])) == 1:
            return True, board[0]
        elif board[3] != 0 and len(set(board[3:6])) == 1:
            return True, board[3]
        elif board[6] != 0 and len(set(board[6:])) == 1:
            return True, board[6]
        elif board[0] != 0 and len(set([board[i] for i in [0,3,6]])) == 1:
            return True, board[0]
        elif board[1] != 0 and len(set([board[i] for i in [1,4,7]])) == 1:
            return True, board[1]
        elif board[2] != 0 and len(set([board[i] for i in [2,5,8]])) == 1:
            return True, board[2]
        elif board[0] != 0 and len(set([board[i] for i in [0,4,8]])) == 1:
            return True, board[0]
        elif board[2] != 0 and len(set([board[i] for i in [2,4,6]])) == 1:
            return True, board[2]
        elif len(self.get_empty_pos()) == 0:
            return True, 0
        return False, 0

    def __str__(self):
        board = ['X' if x == 1 else x for x in self.board]
        board = ['O' if x == -1 else x for x in board]
        board = ['.' if x == 0 else x for x in board]
        return "{b[0]} | {b[1]} | {b[2]}\n--+---+---\n{b[3]} | {b[4]} | {b[5]}\n--+---+---\n{b[6]} | {b[7]} | {b[8]}\n\n".format(b = board)

class TreeNode():
    def __init__(self, parent):
        self._n_visit = 0
        self._Q = 0
        self.children = {}
        self.parent = parent

    def update(self, z):
        if self.parent:
            self.parent.update(z)
        self.back_up(z)

    def back_up(self, z):
        self._n_visit += 1
        self._Q += (z - self._Q) / self._n_visit

    def select(self, board, c, player, print_q = False):
        if board.current_player == player:
            q_value = sorted([(action, node.get_Q(c)) for action, node in self.children.items()], key = lambda x: -x[1])
            if print_q:
                print(q_value)
            return max(self.children.items(), key = lambda x: x[1].get_Q(c))
        else:
            q_value = sorted([(action, node.get_Q(-c)) for action, node in self.children.items()], key = lambda x: x[1])
            if print_q:
                print(q_value)
            return min(self.children.items(), key = lambda x: x[1].get_Q(-c))
            

    def get_Q(self, c):
        return self._Q + c * (np.sqrt(np.log(self.parent._n_visit) / (self._n_visit + 1)))

    def expand(self, board):
        empty_pos = board.get_empty_pos()
        for pos in empty_pos:
            if pos not in self.children:
                self.children[pos] = TreeNode(self)

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS():
    def __init__(self, default_policy_fn, c, seconds = 30):
        self.default_policy_fn = default_policy_fn
        self.calculation_time = datetime.timedelta(seconds = seconds)
        self.c = c
        self.root = TreeNode(None)

    def search(self, board):
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.simulate(board)
        return self.root.select(board, self.c, board.current_player,True)[0]

    def simulate(self, board):
        node = self.root
        current_player = board.current_player
        current_board = copy.deepcopy(board)
        while not current_board.is_over()[0]:
            if node.is_leaf():
                node.expand(current_board)
                break
            action, node = node.select(current_board, self.c, current_player)
            current_board.add_piece(action)
        z = self.evaluate(current_board, current_player)
        node.update(z)

    def evaluate(self, board, player):
        while not board.is_over()[0]:
            action = self.default_policy_fn(board)
            board.add_piece(action)
        _, winner = board.is_over()
        if winner == 0:
            return 0
        return 1 if winner == player else -1

    def update_last_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None)

def random_choice(board):
    empty_pos = board.get_empty_pos()
    return choice(empty_pos)

class MCTSPlayer():
    def __init__(self, seconds):
        self.mcts = MCTS(random_choice, 5, seconds)

    def predict(self, board):
        empty_positions = board.get_empty_pos()
        if len(empty_positions) > 0:
            move = self.mcts.search(board)
            self.mcts.update_last_move(-1)
            return move
        else:
            print("WARNING: the board is full")

def game(player1, player2, print_board = True):
    board = TTTBoard()
    players = [player1, player2]
    current_player_idx = 0
    while not board.is_over()[0]:
        current_player = players[current_player_idx]
        move = current_player.predict(board)
        board.add_piece(move)
        if print_board:
            print(board)
        current_player_idx  = (current_player_idx + 1) % 2


if __name__ == "__main__":
    player1 = MCTSPlayer(30)
    player2 = MCTSPlayer(30)
    
    for i in range(10):
        print("-------start--------")
        game(player1, player2)
        print("-------end:{}--------".format(i))








