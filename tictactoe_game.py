class TicTacToeBoard:
    def __init__(self):
        self.board = [0 for _ in range(9)]

    def start(self):
        self.board = [0 for _ in range(9)]

    def get_board(self):
        return self.board

    def get_empty_pos(self):
        return [index for index, value in enumerate(self.get_board()) if value == 0]

    def add_piece(self, player, move):
        if move in self.get_empty_pos():
            self.board[move] = player

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
if __name__ == "__main__":
    ttt = TicTacToeBoard()
    ttt.add_piece(-1, 4)
    print(ttt)
    print(ttt.is_over())
