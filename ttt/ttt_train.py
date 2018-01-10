from tictactoe_game import TicTacToeBoard
from ttt_agent import TTTPolicyGradientAgent, TTTRandomAgent, TTTGeneticAgent

PG_model_name1 = "model1"
PG_model_name2 = "model2"
GN_model_name = "genomes"

class TTTTrain():
    def __init__(self, episodes, learning_rate):
        self.episodes = episodes
        self.learning_rate = learning_rate

    def play_game(self, p1, p2):
        board = TicTacToeBoard()
        p1_board_memory = []
        p2_board_memory = []
        p1_move_memory = []
        p2_move_memory = []

        for _ in range(5):
            p1_board_memory.append(board.get_board().copy())
            p1_move = p1.next_move(board, print_board = False)
            p1_move_memory.append(p1_move)
            if board.is_over()[0]:
                break

            p2_board_memory.append(board.get_board().copy())
            p2_move = p2.next_move(board, print_board = False)
            p2_move_memory.append(p2_move)
            if board.is_over()[0]:
                break
        winner = board.is_over()[1]
        return winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory

    def train(self, player1, player2):
        for i in range(self.episodes):
            winner, p1_board_memory, p1_move_memory, p2_board_memory, p2_move_memory = self.play_game(player1, player2)
            player1.update_model(1, winner, p1_board_memory, p1_move_memory, self.learning_rate)
            player2.update_model(-1, winner, p2_board_memory, p2_move_memory, self.learning_rate)
            if i % 100 == 0:
                print("Iterate {}#".format(i+1))


class TTTPolicyGradientTrain(TTTTrain):

    def pg_train(self):
        player1 = TTTGeneticAgent(1, GN_model_name)
        player2 = TTTPolicyGradientAgent(-1, PG_model_name2)
        self.train(player1, player2)

        player1 = TTTPolicyGradientAgent(1, PG_model_name1)
        player2 = TTTGeneticAgent(-1, GN_model_name)
        self.train(player1, player2)

    def test(self):
        results = [0, 0, 0]

        agent_player1 = TTTPolicyGradientAgent(1, PG_model_name1)
        random_player1 = TTTRandomAgent(-1)

        agent_player2 = TTTPolicyGradientAgent(-1, PG_model_name2)
        random_player2 = TTTRandomAgent(1)

        for i in range(1000):
            if i < 500:
                s = 1
                winner, _, _, _, _ = self.play_game(agent_player1, random_player1)
            else:
                s = -1
                winner, _, _, _, _ = self.play_game(random_player2, agent_player2)

            results[abs(winner - s)] += 1

        print("win rate: {}\n win - {r[0]} lose - {r[1]} draw - {r[2]}".format(results[0] / sum(results), r=results))
        
class TTTGeneticTrain(TTTTrain):

    def gn_train(self):
        player1 = TTTRandomAgent(1)
        player2 = TTTGeneticAgent(-1, GN_model_name)
        self.train(player1, player2)

        player1 = TTTGeneticAgent(1, GN_model_name)
        player2 = TTTRandomAgent(-1)
        self.train(player1, player2)

        player1 = TTTGeneticAgent(1, GN_model_name)
        player2 = TTTGeneticAgent(-1, GN_model_name)
        self.train(player1, player2)

    def test(self):
        results = [0, 0, 0]

        agent_player1 = TTTGeneticAgent(1, GN_model_name)
        random_player1 = TTTRandomAgent(-1)

        agent_player2 = TTTGeneticAgent(-1, GN_model_name)
        random_player2 = TTTRandomAgent(1)

        for i in range(1000):
            if i < 500:
                s = 1
                winner, _, _, _, _ = self.play_game(agent_player1, random_player1)
            else:
                s = -1
                winner, _, _, _, _ = self.play_game(random_player2, agent_player2)

            results[abs(winner - s)] += 1

        print("win rate: {}\n win - {r[0]} lose - {r[1]} draw - {r[2]}".format(results[0] / sum(results), r=results))
        
if __name__ == "__main__":
    # pg_train = TTTPolicyGradientTrain(10000, 0.02)
    # pg_train.pg_train()
    # pg_train.test()


    gn_train = TTTGeneticTrain(10000, 0.02)
    #gn_train.gn_train()
    gn_train.test()
