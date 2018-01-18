from numpy import random, sqrt, exp, dot, argmax
from random import choices, choice, uniform

class Genome():
    def __init__(self, two_mine_pieces_in_row       = uniform(0, 1),
                       two_opponent_pieces_in_row   = uniform(0, 1),
                       mine_piece_in_open_row       = uniform(0, 1),
                       opponent_piece_in_open_row   = uniform(0, 1),
                       three_mine_pieces_in_row     = uniform(0, 1),
                       three_opponent_pieces_in_row = uniform(0, 1),
                       fitness                      = -1):
        self.two_mine_pieces_in_row       = two_mine_pieces_in_row
        self.two_opponent_pieces_in_row   = two_opponent_pieces_in_row
        self.mine_piece_in_open_row       = mine_piece_in_open_row
        self.opponent_piece_in_open_row   = opponent_piece_in_open_row
        self.three_mine_pieces_in_row     = three_mine_pieces_in_row
        self.three_opponent_pieces_in_row = three_opponent_pieces_in_row
        self.fitness                      = fitness

#train 10 games test 1 games to get fitness for that genomes
class Genetic:
    def __init__(self, genomes, current_genome_idx, population_size):
        self.mutation_rate = 0.2
        self.mutation_step = 0.2

        self.current_genome = current_genome_idx
        self.population_size = population_size
        self.genomes = genomes
        self.evaluate_next_genome()

    def update(self, score):
        self.genomes[self.current_genome].fitness = score
        self.evaluate_next_genome()

    def predict(self, board):
        player = board.current_player
        current_board = board.get_board()
        current_choices = self.predict_one_depth(current_board, player)
        current_player = player
        #depth = 2
        current_player *= (-1)
        for choice in current_choices:
            next_board = choice[3]
            next_choices = self.predict_one_depth(next_board, current_player)
            next_best_move = self.best_choice(next_choices)
            if current_player == player:
                choice[1] += next_best_move[1]
                choice[2] += next_best_move[2]
            else:
                choice[1] += next_best_move[2]
                choice[2] += next_best_move[1]
            choice[3] = next_best_move[3]
        return self.best_choice(current_choices)[0]

    def predict_one_depth(self, board, player):
        valid_choices = [idx for idx in range(len(board)) if board[idx] == 0]
        if len(valid_choices) == 0:
            return [[-1, 0, 0, board]]

        choices = []
        for pos in valid_choices:
            new_board = board[:]
            new_board[pos] = player
            choices += [[pos, self.score(new_board, player), self.score(new_board, player * (-1)) ,new_board]]
        return choices

    def best_choice(self, choices):
        return max(choices, key = lambda x: 0.5 * x[1] + (-0.5) * x[2])

    def score(self, board, player):
        targets = [0 for _ in range(6)]

        for i in range(3):
            for j in range(3):
                for n in [1,2,3,4]:
                    for m in range(1,4):

                        pieces_in_one_line = (i * 3 + j + (m-1) * n) // 3 == ((i * 3 + j) // 3) + ((m-1) if n > 1 else 0)

                        open_line = i * 3 + j - n >= 0 and board[i * 3 + j - n] == 0 and\
                                    i * 3 + j + m * n < 9 and board[i * 3 + j + m * n] == 0

                        same_piece_in_line = i * 3 + j + (m-1) * n < 9 and len(set([board[i * 3 + j + k * n] for k in range(m)])) == 1
        
                        if board[i * 3 + j] !=0 and \
                           pieces_in_one_line and \
                           (open_line if m == 1 else True) and \
                           same_piece_in_line:
                            
                            if board[i * 3 + j] == player:
                                targets[(m - 1) * 2] += 1
                            else:
                                targets[(m - 1) * 2 + 1] += 1
        return targets[0] * self.genomes[self.current_genome].mine_piece_in_open_row +\
               targets[1] * self.genomes[self.current_genome].opponent_piece_in_open_row +\
               targets[2] * self.genomes[self.current_genome].two_mine_pieces_in_row +\
               targets[3] * self.genomes[self.current_genome].two_opponent_pieces_in_row +\
               targets[4] * self.genomes[self.current_genome].three_mine_pieces_in_row +\
               targets[5] * self.genomes[self.current_genome].three_opponent_pieces_in_row        

    def evaluate_next_genome(self):
        self.current_genome += 1
        if self.current_genome >= self.population_size:
            self.evolve()

    def evolve(self):
        self.current_genome = 0
        self.genomes = sorted(self.genomes, key = lambda x: -x.fitness)
        while len(self.genomes) > self.population_size // 2:
            self.genomes.pop()
        children = [self.genomes[0]]
        while len(children) < self.population_size:
            children += [self.make_child(choices(self.genomes, k=2))]
        self.genomes = children

    def make_child(self, parents):
        mum, dad = parents
        
        child = Genome(two_mine_pieces_in_row       = choice([mum.two_mine_pieces_in_row,       dad.two_mine_pieces_in_row]),
                       two_opponent_pieces_in_row   = choice([mum.two_opponent_pieces_in_row,   dad.two_opponent_pieces_in_row]),
                       mine_piece_in_open_row       = choice([mum.mine_piece_in_open_row,       dad.mine_piece_in_open_row]),
                       opponent_piece_in_open_row   = choice([mum.opponent_piece_in_open_row,   dad.opponent_piece_in_open_row]),
                       three_mine_pieces_in_row     = choice([mum.three_mine_pieces_in_row,     dad.three_mine_pieces_in_row]),
                       three_opponent_pieces_in_row = choice([mum.three_opponent_pieces_in_row, dad.three_opponent_pieces_in_row]),)
        if uniform(0, 1) < self.mutation_rate:
            child.two_mine_pieces_in_row       += uniform(0, 1) * self.mutation_step * 2 - self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.two_opponent_pieces_in_row   += uniform(0, 1) * self.mutation_step * 2 - self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.mine_piece_in_open_row       += uniform(0, 1) * self.mutation_step * 2 - self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.opponent_piece_in_open_row   += uniform(0, 1) * self.mutation_step * 2 - self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.three_mine_pieces_in_row     += uniform(0, 1) * self.mutation_step * 2 - self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.three_opponent_pieces_in_row += uniform(0, 1) * self.mutation_step * 2 - self.mutation_step
        return child
