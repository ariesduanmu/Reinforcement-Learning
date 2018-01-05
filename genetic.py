from numpy import random, sqrt, exp, dot, argmax
from random import choices, choice, uniform
import pickle
import os

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
    def __init__(self):
        self.mutation_rate = 0.2
        self.mutation_step = 0.2

        self.current_genome = -1
        self.population_size = 10
        self.genomes = []
        self.read_dataset()
        self.evaluate_next_genome()

    def update(self, score):
        self.genomes[self.current_genome].fitness = score
        self.evaluate_next_genome()

    def predict(self, grid, player):
        valid_choice = [(i, j) for i in range(3) for j in range(3) if grid[i][j] == 0]
        scores = []
        for pos in valid_choice:
            g = [row[:] for row in grid]
            g[pos[0]][pos[1]] = 1
            scores += [[pos, self.score(g,player)]]
        return sorted(scores, key = lambda x: -x[1])[0][0]


    def score(self, grid, player):
        score_ = 0
        rotated_grid = [row for row in zip(*grid)]
        def three_in_row():
            def check_h_v(a):
                for r in a:
                    if sum(r) == 3:
                        return 3
                    elif sum(r) == -3:
                        return -3
                return 0
            h = check_h_v(grid)
            v = check_h_v(rotated_grid)
            a_1 = grid[0][0] + grid[1][1] + grid[2][2]
            a_2 = grid[0][2] + grid[1][1] + grid[2][0]

            if 3 in [h,v,a_1,a_2]:
                return 1
            elif -3 in [h,v,a_1,a_2]:
                return -1
            return 0
        three = three_in_row()
        if three == player:
            score_ += self.genomes[self.current_genome].three_mine_pieces_in_row
        elif three == -1 * player:
            score_ += self.genomes[self.current_genome].three_opponent_pieces_in_row

        two_m = 0
        two_o = 0
        for r in grid:
            if r.count(player) == 2:
                two_m += 1
            elif r.count(-1 * player) == 2:
                two_o += 1

        for r in rotated_grid:
            if r.count(player) == 2:
                two_m += 1
            elif r.count(-1 * player) == 2:
                two_o += 1

        for r in [[grid[0][0], grid[1][1], grid[2][2]], [grid[0][2], grid[1][1], grid[2][0]]]:
            if r.count(player) == 2:
                two_m += 1
            elif r.count(-1 * player) == 2:
                two_o += 1

        one_m = grid.count([0,0,player]) + grid.count([0,player,0]) + grid.count([player,0,0])\
                + rotated_grid.count([0,0,player]) + rotated_grid.count([0,player,0]) + rotated_grid.count([player,0,0])\
                + 1 if [grid[0][0], grid[1][1], grid[2][2]] in [[0,0,player],[0,player,0],[player,0,0]] else 0\
                + 1 if [grid[0][2], grid[1][1], grid[2][0]] in [[0,0,player],[0,player,0],[player,0,0]] else 0

        one_o = grid.count([0,0,-1 * player]) + grid.count([0,-1 * player,0]) + grid.count([-1 * player,0,0])\
                + rotated_grid.count([0,-1 * player,-1 * player]) + rotated_grid.count([-1 * player,-1 * player,0]) + rotated_grid.count([-1 * player,0,0])\
                + 1 if [grid[0][0], grid[1][1], grid[2][2]] in [[0,0,-1 * player],[0,-1 * player,0],[-1 * player,0,0]] else 0\
                + 1 if [grid[0][2], grid[1][1], grid[2][0]] in [[0,0,-1 * player],[0,-1 * player,0],[-1 * player,0,0]] else 0
        score_ += two_m * self.genomes[self.current_genome].two_mine_pieces_in_row
        score_ += two_o * self.genomes[self.current_genome].two_opponent_pieces_in_row
        score_ += one_m * self.genomes[self.current_genome].mine_piece_in_open_row
        score_ += one_o * self.genomes[self.current_genome].opponent_piece_in_open_row

        return score_


        

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

    def save_dataset(self):
        with open('genomes', 'wb+') as f:
            pickle.dump((self.genomes, self.current_genome), f, -1)
    def read_dataset(self):
        if not os.path.isfile('genomes'):
            self.genomes = [Genome() for _ in range(self.population_size)]
            
        else:
            with open('genomes', 'rb') as f:
                self.genomes, self.current_genome = pickle.load(f)
