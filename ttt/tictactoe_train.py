from random import choice, randint
from numpy import array, reshape
import pickle
from genetic import Genetic
import os
import pdb

MAX_train = 500000

class TictactoeTrain():
    def __init__(self):
        self.game_count = 0
        self.new_game()

    def random_train(self):
        self.memories = []
        self.all_memories = {}
        self.read_dataset()
        while self.game_count <= MAX_train:
            self.random_play_train()
            finish, winner = self.check_game_finish()
            if finish:
                if winner == 1:
                    self.memories = [[memory[0], memory[1], 0 if memory[2] >= 0 else -1] for memory in self.memories ]
                elif winner == -1:
                    self.memories = [[memory[0], memory[1], 1 if memory[2] >= 0 else -1] for memory in self.memories ]
                else:
                    self.memories = [[memory[0], memory[1], 0.2 if memory[2] >= 0 else -1] for memory in self.memories]

                for memory in self.memories:
                    g = memory[0]
                    c = ' '.join(map(str,memory[1]))
                    if g not in self.all_memories:
                        if memory[2] < 0:
                            self.all_memories[g] = {c:{"win_rate":-1}}
                        else:
                            self.all_memories[g] = {c:{"win_rate":memory[2],"game":1}}

                    elif c not in self.all_memories[g]:
                        if memory[2] < 0:
                            self.all_memories[g][c] = {"win_rate":-1}
                        else:
                            self.all_memories[g][c] = {"win_rate":memory[2],"game":1}
                        
                    else:
                        r = self.all_memories[g][c]
                        
                        if r["win_rate"] >= 0 and memory[2] >= 0:
                            r["win_rate"] = (r["win_rate"] * r["game"] + memory[2]) / (r["game"] + 1)
                            r["game"] += 1
                        
                        self.all_memories[g][c] = r

                

                self.memories = []
                self.save_dataset()
                if self.game_count % 1000 == 0:
                    print("Current process: {}".format(self.game_count))
                self.new_game()


    def test_random_train(self):
        self.win1_ = 0
        self.loss1_ = 0
        self.draw1_ = 0
        self.error1_ = 0

        self.win2_ = 0
        self.loss2_ = 0
        self.draw2_ = 0
        self.error2_ = 0

        self.all_memories = {}
        self.read_dataset()

        self.in_ = 0
        self.moves = 0
        while True:
            if self.player == 1:
                self.random_player()
                #self.ai_random_player1()
            else:
                self.ai_random_player2()
            finish, winner = self.check_game_finish()
            if finish:
                if winner == 1:
                    self.loss2_ += 1
                    #self.win1_ += 1
                elif winner == -1:
                    self.win2_ += 1
                    #self.loss1_ += 1
                else:
                    self.draw2_ += 1
                    #self.draw1_ += 1

                if self.game_count > 0 and self.game_count % 10000 == 0:
                    w2 = self.win2_ / (self.win2_ + self.loss2_ + self.draw2_ + self.error2_)
                    d2 = self.draw2_ / (self.win2_ + self.loss2_ + self.draw2_ + self.error2_)
                    l2 = self.loss2_ / (self.win2_ + self.loss2_ + self.draw2_ + self.error2_)
                    e2 = self.error2_ / (self.win2_ + self.loss2_ + self.draw2_ + self.error2_)
                    #w1 = self.win1_ / (self.win1_ + self.loss1_ + self.draw1_ + self.error1_)
                    #d1 = self.draw1_ / (self.win1_ + self.loss1_ + self.draw1_ + self.error1_)
                    #l1 = self.loss1_ / (self.win1_ + self.loss1_ + self.draw1_ + self.error1_)
                    #e1 = self.error1_ / (self.win1_ + self.loss1_ + self.draw1_ + self.error1_)
                    #print("win1: {}, draw1: {}, loss1: {}, error1: {}".format(w1,d1,l1,e1))
                    print("win2: {}, draw2: {}, loss2: {}, error2: {}, in: {}".format(w2,d2,l2,e2, self.in_ / self.moves))
                  
                self.new_game()


    def neural_train(self, model):
        self.score = 0
        self.model = model
        self.scores = []
        while True:
            if self.player == 1:
                self.random_player()
            else:
                self.ai_neural_player()
            finish, winner = self.check_game_finish()
            if finish:
                if winner == 1:
                    self.score -= 5
                elif winner == -1:
                    self.score += 10
                else:
                    self.score += 2

                if self.game_count > 0 and self.game_count % 1000 == 0:
                    self.model.update(self.score)
                    self.model.save_dataset()
                    print(len([s for s in self.scores if s > 2]) / len(self.scores))
                    self.scores = []

                self.scores.append(self.score)
                self.score = 0
                self.new_game()

    def random_play_train(self):
        g = ' '.join(' '.join(map(str,m)) for m in self.grid)
        if self.player == -1:
            bad_choices = [(i, j) for i in range(3) for j in range(3) if self.grid[i][j] != 0]
            for c in bad_choices:
                self.memories += [[g, c, -1]]

        choices = [(i, j) for i in range(3) for j in range(3) if self.grid[i][j] == 0]
        pos = choice(choices)
        
        #pdb.set_trace()
        if self.add_piece(pos):
            if self.player == -1:
                self.memories += [[g, pos, 0]]
            


    def random_player(self):
        choices = [(i, j) for i in range(3) for j in range(3) if self.grid[i][j] == 0]
        _ = self.add_piece(choice(choices))
    
    def ai_random_player1(self):
        move = self.predict(self.grid)
        if not self.add_piece(move):
            self.error1_ += 1
            self.new_game()

    def ai_random_player2(self):
        move = self.predict(self.grid)
        if not self.add_piece(move):
            #print(self.grid)
            #print(move)
            self.error2_ += 1
            self.new_game()


    def ai_neural_player(self):
        move = self.model.predict(self.grid, -1)
        if not self.add_piece(move):
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
        #pdb.set_trace()
        self.moves += 1
        if g in self.all_memories:
            best_choice = sorted([[c, self.all_memories[g][c]["win_rate"]] for c in self.all_memories[g]], key = lambda x: -x[1])
            
            if best_choice[0][1] > 0:
                self.in_ += 1
                return list(map(int,best_choice[0][0].split(' ')))
        return choice([(i,j) for i in range(3) for j in range(3) if grid[i][j] == 0])

    def save_dataset(self):
        with open('randomTrain', 'wb+') as f:
            pickle.dump((self.all_memories), f, -1)

    def read_dataset(self):
        if os.path.isfile('randomTrain'):
            with open('randomTrain', 'rb') as f:
                self.all_memories = pickle.load(f)


def main():
    model = Genetic()
    game = TictactoeTrain()
    #game.random_train()
    #game.test_random_train()
    game.neural_train(model)

if __name__ == "__main__":
    main()
