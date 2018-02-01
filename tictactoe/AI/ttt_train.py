from ttt_game import TTTGame
from ttt_agent import TTTPolicyGradientAgent, TTTRandomAgent, TTTGeneticAgent, TTTMCTSAgent

PG_model_name1 = "model1"
PG_model_name2 = "model2"
GN_model_name = "genomes"

class TTTTrain():
    def __init__(self, episodes = 10000, learning_rate = 0.02):
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.evaluate_fequence = 100
        self.best_win_ration = 0
        self.game = TTTGame()

    def run(self, player1, player2):
        winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory = self.game.play(player1, player2)
        player1.update_model(1, winner, p1_board_memory, p1_move_memory, self.learning_rate)
        player2.update_model(-1, winner, p2_board_memory, p2_move_memory, self.learning_rate)

    def evaluate_train(self, agent_player1_1, agent_player2_2, agent_player1_2 = TTTRandomAgent(), agent_player2_1 = TTTRandomAgent(), print_board = False):
        results = [0, 0, 0]

        for i in range(1000):
            if i < 500:
                s = -1
                winner, _, _, _, _ = self.game.play(agent_player2_1, agent_player2_2, print_board)
            else:
                s = 1
                winner, _, _, _, _ = self.game.play(agent_player1_1, agent_player1_2, print_board)

            results[abs(winner - s)] += 1

            if i % 5 == 0:
                win_ratio = results[0] / sum(results)
                print("win rate: {}\n win - {r[0]} lose - {r[2]} draw - {r[1]}".format(win_ratio, r=results))

        win_ratio = results[0] / sum(results)
        print("win rate: {}\n win - {r[0]} lose - {r[2]} draw - {r[1]}".format(win_ratio, r=results))
        return win_ratio

class TTTPolicyGradientTrain(TTTTrain):

    def train(self):
        for i in range(self.episodes):
            player1 = TTTGeneticAgent(GN_model_name)
            player2 = TTTPolicyGradientAgent(PG_model_name2)
            self.run(player1, player2)

            player1 = TTTPolicyGradientAgent(PG_model_name1)
            player2 = TTTGeneticAgent(GN_model_name)
            self.run(player1, player2)

            if i % self.evaluate_fequence == 0:
                print("Iterate {}#".format(i+1))
                win_ratio = self.evaluate()
                if win_ratio > self.best_win_ration:
                    print("[+] New best policy")
                    self.best_win_ration = win_ratio
                    #pickle.dump(data, open('best_policy_grandient.model', 'wb+'))
    
    def evaluate(self):
        agent_player1 = TTTPolicyGradientAgent(PG_model_name1)
        agent_player2 = TTTPolicyGradientAgent(PG_model_name2)

        return self.evaluate_train(agent_player1, agent_player2)
    

class TTTGeneticTrain(TTTTrain):

    def train(self):
        for i in range(self.episodes):
            player1 = TTTRandomAgent()
            player2 = TTTGeneticAgent(GN_model_name)
            self.run(player1, player2)

            player1 = TTTGeneticAgent(GN_model_name)
            player2 = TTTRandomAgent()
            self.run(player1, player2)

            player1 = TTTGeneticAgent(GN_model_name)
            player2 = TTTGeneticAgent(GN_model_name)
            self.run(player1, player2)

            if i % self.evaluate_fequence == 0:
                print("Iterate {}#".format(i+1))
                win_ratio = self.evaluate()
                if win_ratio > self.best_win_ration:
                    print("[+] New best gene")
                    self.best_win_ration = win_ratio
                    #pickle.dump(data, open('best_gene.model', 'wb+'))
    def get_lots_data(self):
        winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory = self.game.play(agent_player2_1, agent_player2_2, print_board)

    def evaluate(self):
        agent_player1 = TTTGeneticAgent(GN_model_name)
        agent_player2 = TTTGeneticAgent(GN_model_name)
        
        return self.evaluate_train(agent_player1, agent_player2)

class TTTMCTSTrain(TTTTrain):
    def evaluate(self):
        for i in range(10):
            agent_player2_2 = TTTGeneticAgent(GN_model_name)
            agent_player2_1 = TTTMCTSAgent(PG_model_name1)

            winner, _, _, _, _ = self.game.play(agent_player2_1, agent_player2_2, True)
            print("---------winner: {} * {}/10---------".format(winner, i))

if __name__ == "__main__":
    # pg_train = TTTPolicyGradientTrain(10000, 0.02)
    # pg_train.train()
    #gn_train = TTTGeneticTrain(10000, 0.02)
    #gn_train.train()

    mct_train = TTTMCTSTrain()
    mct_train.evaluate()
