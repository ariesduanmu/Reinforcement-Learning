import pygame
from numpy import array, reshape
from tictactoe_model import TttModel
import time
from random import choice
SCREEN_WIDTH = 180

WHITE = (255,255,255)
GRAY = (150,150,150)

RED = (133,42,44)
GREEN = (26,81,79)

FILE_NAME = "Tictactoe_weight.txt"

class Tictactoe(object):
	def __init__(self, model ,player=1):
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH),pygame.HWSURFACE | pygame.DOUBLEBUF)
		pygame.display.set_caption('Tic-Tac-Toe')
		
		self.first_player = player
		self.reset_game()

		self.model = model

		self.__computer_memory = []
		self.__human_memory = []


	def test(self):
		if self.__player == 1:
			for event in pygame.event.get():
				self.on_event(event)
		else:
			self.computer_player()


	def on_execute(self):
		while self.__running:
			self.game_board_init()
			self.test()
			
			self.on_render()
			self.check_game_finish()
			
		self.on_cleanup()


	def on_event(self, event):
		if event.type == pygame.QUIT:
			self.__running = False
		self.human_player(event)

	def on_render(self):
		self.render_piece()
		pygame.display.update()

	def on_cleanup(self):
		pygame.quit()

	def check_game_finish(self):

		finish = True
		if self.check_win() != 0:
			
			if self.__player == -1:
				self.__human_memory = [[g,c*(-0.1)] for g,c in self.__human_memory]
			else:
				self.__computer_memory = [[g,c*(-0.1)] for g,c in self.__computer_memory]
			
			winner = self.check_win()
		elif self.check_full():
			self.__human_memory = [[g,c*(0.1)] for g,c in self.__human_memory]
			self.__computer_memory = [[g,c*(0.1)] for g,c in self.__computer_memory]
			winner = 0
		else:
			finish = False

		if finish:
			# train computer player
			if self.model is not None:
				replay_memory = self.__computer_memory + self.__human_memory
				self.model.train_network(replay_memory)
			if winner == 0:
				print('draw')
			elif winner == -1:
				print('winner is -1')
			elif winner == 1:
				print('winner is 1')
			time.sleep(1)
			self.reset_game()

	def reset_game(self):
		self.grid = [[0 for _ in range(3)] for _ in range(3)]
		self.__running = True
		self.__player = self.first_player

	def computer_player(self):
		if self.model is not None:
			cur_grid = reshape(array(self.grid),(1,9))[0]
			piece_pos_chosen = self.model.predict(cur_grid)
			self.add_piece(piece_pos_chosen)

	def human_player(self, event):
		if event.type == pygame.MOUSEBUTTONUP:
			pos = pygame.mouse.get_pos()
			row_height = SCREEN_WIDTH // 3

			col = pos[0] // row_height
			row = pos[1] // row_height
			
			self.add_piece([row, col])
	
	def random_play(self):
		choices = [(i, j) for i in range(3) for j in range(3)]
		self.add_piece(choice(choices))

	def add_piece(self, pos):
		
		choice = [0 for _ in range(9)]
		choice[pos[0] * 3 + pos[1]] = 1
		choice = array([choice])

		cur_grid = reshape(array(self.grid),(1,9))
		

		if self.validate_position(pos):
			
			if self.__player == -1:
				self.__computer_memory.append([cur_grid, choice])
			else:
				self.__human_memory.append([cur_grid * -1,choice])

			self.__player = -1 * self.__player
		else:
			if self.__player == -1:
				self.model.train_network([[cur_grid, choice * -1000]])
			else:
				
				self.model.train_network([[cur_grid * -1, choice * -1000]])
			print('error position: other pieces already in here')


	def render_piece(self):
		width = SCREEN_WIDTH // 3
		margin = 10
		for i in range(3):
			for j in range(3):
				center = ((width // 2) + j * width, (width // 2) + i * width)
				if self.grid[i][j] != 0:
					color = RED if self.grid[i][j] == 1 else GREEN
					pygame.draw.circle(self.screen, color,
                                       center,
                                       width // 2 - margin, 0)

	def game_board_init(self):
		self.screen.fill(GRAY)

		width = SCREEN_WIDTH // 3
		margin = 4
		for i in range(3):
			for j in range(3):
				pygame.draw.rect(self.screen, WHITE,
                                 [width * j + margin // 2,
                                  width * i + margin // 2,
                                  width - margin,
                                  width - margin])
	def validate_position(self, pos):

		if self.grid[pos[0]][pos[1]] == 0:
			self.grid[pos[0]][pos[1]] = self.__player
			return True
		return False
	def check_win(self):
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
	def check_full(self):
		for r in self.grid:
			if 0 in r:
				return False
		return True


def main():
	model = TttModel(FILE_NAME)
	game = Tictactoe(model = model)
	game.on_execute()
	

if __name__ == '__main__':
	main()