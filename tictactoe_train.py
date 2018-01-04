from random import choice, randint
from numpy import array, reshape
import pickle
from genetic import Genetic
import os

MAX_train = 50000

class TictactoeTrain():
    def __init__(self):
        self.game_count = 0
        self.new_game()

    def random_train(self):
        self.memories = []
        self.all_memories = {}
        self.read_dataset()
        while True:
            self.random_play_train()
            finish, winner = self.check_game_finish()
            if finish:
                if winner == 1:
                    self.memories = [[memory[0], memory[1], -1] for memory in self.memories]
                elif winner == -1:
                    self.memories = [[memory[0], memory[1], 1] for memory in self.memories]
                else:
                    self.memories = [[memory[0], memory[1], 0.2] for memory in self.memories]

                for memory in self.memories:
                    g = ' '.join(' '.join(map(str,m)) for m in memory[0])
                    c = ' '.join(map(str,memory[1]))
                    if g not in self.all_memories or c not in self.all_memories[g]:
                        self.all_memories[g] = {c:memory[2]}
                    else:
                        self.all_memories[g][c] += memory[2]
                self.memories = []
                self.save_dataset()
                if self.game_count % 1000 == 0:
                    print("Current process: {}".format(self.game_count))
                self.new_game()


    def test_random_train(self):
        self.win_ = 0
        self.loss_ = 0
        self.draw_ = 0
        self.error_ = 0

        self.all_memories = {}
        self.read_dataset()
        while True:
            if self.player == 1:
                self.random_player()
            else:
                self.ai_random_player()
            finish, winner = self.check_game_finish()
            if finish:
                if winner == 1:
                    self.loss_ += 1
                elif winner == -1:
                    self.win_ += 1
                else:
                    self.draw_ += 1

                if self.game_count > 0 and self.game_count % 10000 == 0:
                    w = self.win_ / (self.win_ + self.loss_ + self.draw_ + self.error_)
                    d = self.draw_ / (self.win_ + self.loss_ + self.draw_ + self.error_)
                    l = self.loss_ / (self.win_ + self.loss_ + self.draw_ + self.error_)
                    e = self.error_ / (self.win_ + self.loss_ + self.draw_ + self.error_)
                    print("win: {}, draw: {}, loss: {}, error: {}".format(w,d,l,e))
                    
                self.new_game()

    def neural_train(self, model):
        self.score = 0
        self.model = model
        while True:
            if self.player == 1:
                self.random_player()
            else:
                self.ai_neural_player()
            finish, winner = self.check_game_finish()
            if finish:
                if winner == 1:
                    self.score += 1
                elif winner == -1:
                    self.score += 10
                else:
                    self.score += 10

                if self.game_count > 0 and self.game_count % 1000 == 0:
                    self.score = 0
                    #self.model.update(self.score)
                    #self.model.save_dataset()
                    #self.validTest()

                self.new_game()

    def random_play_train(self):
        if self.player == -1:
            bad_choices = [(i, j) for i in range(3) for j in range(3) if self.grid[i][j] == 0]
            for c in bad_choices:
                self.memories += [[self.grid, c, -10]]

        choices = [(i, j) for i in range(3) for j in range(3) if self.grid[i][j] == 0]
        pos = choice(choices)
        if self.add_piece(pos):
            if self.player == -1:
                self.memories += [[self.grid, pos, 0]]


    def random_player(self):
        choices = [(i, j) for i in range(3) for j in range(3) if self.grid[i][j] == 0]
        _ = self.add_piece(choice(choices))
    
    def ai_random_player(self):
        move = self.predict(self.grid)
        if not self.add_piece(move):
            self.error_ += 1
            self.new_game()


    def ai_neural_player(self):
        # cur_choice = [0 for _ in range(9)]
        # cur_choice[pos[0] * 3 + pos[1]] = 1
        # cur_choice = array([cur_choice])
        # cur_grid = reshape(array(self.grid),(1,9))
        move = self.model.predict(reshape(array(self.grid),(1,9))[0])
        if self.add_piece(move):
            self.score += 1
        else:
            self.score -= 100
            self.new_game()

    def check_game_finish(self):
        winner = self.has_winner()
        finish = False
        if winner != 0:
            finish = True
            #print("winner is: {}".format(winner))
        else:
            if self.is_full():
                finish = True
                #print("draw")
        #self.grid_output()
        return finish, winner

    def validTest(self):
        error = []
        correct = []
        for _ in range(10):
            player1_num = randint(1,4)
            player2_num = player1_num + choice([0, -1])
            
            grid = [0 for _ in range(9)]

            for _ in range(player1_num):
                choices = [i for i in range(9) if grid[i] == 0]
                
                grid[choice(choices)] = 1

            for _ in range(player2_num):
                choices = [i for i in range(9) if grid[i] == 0]
                
                grid[choice(choices)] = -1

            grid = array(grid)
            pre = self.model.predict(grid)
            row, col = pre
            if grid[row * 3 + col] == 0:
                correct += [pre]
            else:
                error += [pre]
        print("Valid: {} Error: {}".format(len(correct), len(error)))


    def grid_output(self):
        print("Trained {} times".format(self.game_count))
        print('\n'.join(' | '.join(map(str,line)) for line in self.grid))


    def new_game(self):
        self.game_count += 1
        self.winner = 0
        self.player = choice([-1,1])
        self.grid = [[0 for _ in range(3)] for _ in range(3)]

    def add_piece(self, pos):
        if self.valid_position(pos):                
            self.player = -1 * self.player
            return True
        return False

    def valid_position(self, pos):
        i, j = pos
        if self.grid[i][j] == 0:
            self.grid[i][j] = self.player
            return True
        return False

    def has_winner(self):
        def check_h_v(a):
            for r in a:
                if sum(r) == 3:
                    return 3
                elif sum(r) == -3:
                    return -3
            return 0
        g = [[self.grid[j][i] for j in range(3)] for i in range(3)]
        h = check_h_v(self.grid)
        v = check_h_v(g)
        a_1 = self.grid[0][0] + self.grid[1][1] + self.grid[2][2]
        a_2 = self.grid[0][2] + self.grid[1][1] + self.grid[2][0]

        if 3 in [h,v,a_1,a_2]:
            return 1
        elif -3 in [h,v,a_1,a_2]:
            return -1
        return 0

    def is_full(self):
        for r in self.grid:
            if 0 in r:
                return False
        return True

    def predict(self, grid):
        g = ' '.join(' '.join(map(str,d)) for d in grid)
        if g in self.all_memories:
            best_choice = sorted([[c, self.all_memories[g][c]] for c in self.all_memories[g]], key = lambda x: -x[1])[0][0]
            return list(map(int,best_choice.split(' ')))
        return choice([(i,j) for i in range(3) for j in range(3) if grid[i][j] == 0])

    def save_dataset(self):
        with open('randomTrain', 'wb+') as f:
            pickle.dump((self.all_memories), f, -1)

    def read_dataset(self):
        if os.path.isfile('randomTrain'):
            with open('randomTrain', 'rb') as f:
                self.all_memories = pickle.load(f)


def main():
    #model = Genetic()
    game = TictactoeTrain()
    #game.random_train()
    game.test_random_train()

if __name__ == "__main__":
    main()
