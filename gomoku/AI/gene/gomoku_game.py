board_size = 6
class Gomoku():
    def __init__(self):
        self.board = [0 for _ in range(board_size ** 2)]

    def restart(self):
        self.board = [0 for _ in range(board_size ** 2)]

    def get_board(self):
        return self.board

    def get_empty_pos(self):
        return [index for index, value in enumerate(self.get_board()) if value == 0]

    def add_piece(self, player, move):
        if move in self.get_empty_pos():
            self.board[move] = player

    def is_over(self):
        board = self.get_board()

        for i in range(board_size):
            for j in range(board_size):
                for n in [1,board_size-1,board_size,board_size+1]:

                    pieces_in_one_line = (i * board_size + j + 4 * n) // board_size == ((i * board_size + j) // board_size) + (4 if n > 1 else 0)
                    same_piece_in_line = i * board_size + j + 4 * n < board_size ** 2 and len(set([board[i * board_size + j + k * n] for k in range(5)])) == 1

                    if board[i * board_size + j] != 0 and \
                       pieces_in_one_line and \
                       same_piece_in_line:
                        
                        return True, board[i * board_size + j]
        if len(self.get_empty_pos()) == 0:
            return True, 0
        return False, 0

    def __str__(self):
        board = ['X' if x == 1 else x for x in self.board]
        board = ['O' if x == -1 else x for x in board]
        board = ['.' if x == 0 else x for x in board]
        return '\n'.join(' | '.join(board[i * board_size + j] for j in range(board_size)) for i in range(board_size))
