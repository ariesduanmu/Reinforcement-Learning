import numpy as np
import datetime
import copy
from random import choice
from ttt_game import TTTBoard

class TreeNode():
    def __init__(self, parent, prob = 1.0):
        self._n_visit = 0
        self._Q = 0
        self._children = {}
        self._parent = parent
        self._P = prob

    def select(self, board, c, player, print_q = False):
        if board.current_player == player:
            q_value = sorted([(action, node._get_Q(c)) for action, node in self._children.items()], key = lambda x: -x[1])
            if print_q:
                print(q_value)
            return max(self._children.items(), key = lambda x: x[1]._get_Q(c))
        else:
            return min(self._children.items(), key = lambda x: x[1]._get_Q(-c))
    
    def update(self, z):
        if self._parent:
            self._parent.update(z)
        self._back_up(z)

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


    def _back_up(self, z):
        self._n_visit += 1
        self._Q += (z - self._Q) / self._n_visit

    def _get_Q(self, c):
        return self._Q + c * self._P * (np.sqrt(np.log(self._parent._n_visit) / (self._n_visit + 1)))
  

class MCTS():
    def __init__(self, action_policy_fn, default_policy_fn, c, seconds = 30):
        self.default_policy_fn = default_policy_fn
        self.action_policy_fn = action_policy_fn
        self.calculation_time = datetime.timedelta(seconds = seconds)
        self.c = c
        self.root = TreeNode(None, 1.0)

    def search(self, board):
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self._simulate(board)
        return self.root.select(board, self.c, board.current_player)[0]

    def update_last_move(self, last_move):
        if last_move in self.root._children:
            self.root = self.root._children[last_move]
            self.root._parent = None
        else:
            self.root = TreeNode(None, 1.0)
            
    def _simulate(self, board):
        node = self.root
        current_player = board.current_player
        current_board = copy.deepcopy(board)
        while not current_board.is_over()[0]:
            if node.is_leaf():
                action_probs = self.action_policy_fn(current_board)
                node.expand(action_probs)
                break
            action, node = node.select(current_board, self.c, current_player)
            current_board.add_piece(action)
        z = self._evaluate(current_board, current_player)
        node.update(z)

    def _evaluate(self, board, player):
        while not board.is_over()[0]:
            action = self.default_policy_fn(board)
            board.add_piece(action)
        _, winner = board.is_over()
        if winner == 0:
            return 0
        return 1 if winner == player else -1


