from ttt_game import TTTGame
from ttt_agent import TTTPolicyGradientAgent, TTTRandomAgent, TTTGeneticAgent

PG_model_name1 = "model1"
PG_model_name2 = "model2"
GN_model_name = "genomes"

class TTTTrain():
    def __init__(self, episodes, learning_rate):
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.evaluate_fequence = 100
        self.best_win_ration = 0
        self.game = TTTGame()

    def run(self, player1, player2):
        winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory = self.game.play(player1, player2)
        player1.update_model(1, winner, p1_board_memory, p1_move_memory, self.learning_rate)
        player2.update_model(-1, winner, p2_board_memory, p2_move_memory, self.learning_rate)

    def evaluate_train(self, agent_player1, agent_player2):
        results = [0, 0, 0]

        random_player1 = TTTRandomAgent(-1)
        random_player2 = TTTRandomAgent(1)

        for i in range(1000):
            if i < 500:
                s = 1
                winner, _, _, _, _ = self.play_game(agent_player1, random_player1)
            else:
                s = -1
                winner, _, _, _, _ = self.play_game(random_player2, agent_player2)

            results[abs(winner - s)] += 1

        win_ratio = results[0] / sum(results)
        print("win rate: {}\n win - {r[0]} lose - {r[1]} draw - {r[2]}".format(win_ratio, r=results))
        return win_ratio

class TTTPolicyGradientTrain(TTTTrain):

    def train(self):
        for i in range(self.episodes):
            player1 = TTTGeneticAgent(1, GN_model_name)
            player2 = TTTPolicyGradientAgent(-1, PG_model_name2)
            self.run(player1, player2)

            player1 = TTTPolicyGradientAgent(1, PG_model_name1)
            player2 = TTTGeneticAgent(-1, GN_model_name)
            self.run(player1, player2)

            if i % self.evaluate_fequence == 0:
                print("Iterate {}#".format(i+1))
                win_ratio = self.evaluate()
                if win_ratio > self.best_win_ration:
                    print("[+] New best policy")
                    self.best_win_ration = win_ratio
                    #pickle.dump(data, open('best_policy_grandient.model', 'wb+'))
    
    def evaluate(self):
        agent_player1 = TTTPolicyGradientAgent(1, PG_model_name1)
        agent_player2 = TTTPolicyGradientAgent(-1, PG_model_name2)

        return self.evaluate_train(agent_player1, agent_player2)
    

class TTTGeneticTrain(TTTTrain):

    def train(self):
        for i in range(self.episodes):
            player1 = TTTRandomAgent(1)
            player2 = TTTGeneticAgent(-1, GN_model_name)
            self.run(player1, player2)

            player1 = TTTGeneticAgent(1, GN_model_name)
            player2 = TTTRandomAgent(-1)
            self.run(player1, player2)

            player1 = TTTGeneticAgent(1, GN_model_name)
            player2 = TTTGeneticAgent(-1, GN_model_name)
            self.run(player1, player2)

            if i % self.evaluate_fequence == 0:
                print("Iterate {}#".format(i+1))
                win_ratio = self.evaluate()
                if win_ratio > self.best_win_ration:
                    print("[+] New best gene")
                    self.best_win_ration = win_ratio
                    #pickle.dump(data, open('best_gene.model', 'wb+'))

    def evaluate(self):
        agent_player1 = TTTGeneticAgent(1, GN_model_name)
        agent_player2 = TTTGeneticAgent(-1, GN_model_name)
        
        return self.evaluate_train(agent_player1, agent_player2)
        
if __name__ == "__main__":
    # pg_train = TTTPolicyGradientTrain(10000, 0.02)
    # pg_train.train()


    gn_train = TTTGeneticTrain(10000, 0.02)
    #gn_train.train()
