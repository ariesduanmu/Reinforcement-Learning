import numpy as np
import copy
from random import choice
from operator import itemgetter

class TreeNode():
    def __init__(self, parent, prob):
        self._parent = parent
        self._visit = 0
        self._Q = 0
        self._children = {}
        self._P = prob

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c):
        return max(self._children.items(), key=lambda x: x[1].get_Q(c))

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-1 * leaf_value)
        self.update(leaf_value)

    def update(self, leaf_value):
        self._visit += 1
        self._Q += (leaf_value - self._Q) / self._visit

    def get_Q(self, c):
        return self._Q + c * self._P * (np.sqrt(np.log2(self._parent._visit) / (self._visit + 1)))

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent == None

class MCTS():
    def __init__(self, policy_fn, c = 5, max_moves = 100):
        self.c = c
        self.max_moves = max_moves
        self.policy_fn = policy_fn
        self._root = TreeNode(None, 1.0)

    def predict(self, board):
        for i in range(self.max_moves):
            current_board = copy.deepcopy(board)
            self.simulate(current_board)
        return max(self._root._children.items(), key = lambda x: x[1]._visit)[0]

    def simulate(self, board):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c)
            board.add_piece(action)

        action_probs,_ = self.policy_fn(board)
        end, winner = board.is_over()
        if not end:
            node.expand(action_probs)
        leaf_value = self._evaluate(board)
        node.update_recursive(-leaf_value)

    def _evaluate(select, board, limit = 1000):
        player = board.current_player
        for i in range(limit):
            end, winner = board.is_over()
            if end:
                break
            action_probs = select.random_policy(board)
            best_action = max(action_probs, key=itemgetter(1))[0]
            board.add_piece(best_action)
        else:
            print("WARNING: rollout reached move limit")

        if winner == 0:
            return 0
        else:
            return 1 if winner == player else -1

    def random_policy(self, board):
        action_probs = np.random.rand(len(board.get_empty_pos()))
        return zip(board.get_empty_pos(), action_probs)

    def update_last_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
