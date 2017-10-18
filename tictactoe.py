import pygame

SCREEN_WIDTH = 180

WHITE = (255,255,255)
GRAY = (150,150,150)

RED = (133,42,44)
GREEN = (26,81,79)

# human play 1
# computer play -1
class Tictactoe(object):
	def __init__(self, model ,player=1):
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH),pygame.HWSURFACE | pygame.DOUBLEBUF)
		pygame.display.set_caption('Tic-Tac-Toe')
		
		
		self.first_player = player
		self.reset_game()

		self.model = model

		#two part in replay_memory, human part and computer part
		self.replay_memory = []

	def on_execute(self):
		while self.__running:
			self.game_board_init()
			if self.__player == 1:
				for event in pygame.event.get():
					self.on_event(event)
			else:
				self.computer_player()
			
			if self.check_full():
				self.finish_game(0)
			self.on_render()
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

	def finish_game(self, winner):
		# train computer player
		if winner == 0:
			print('draw')
		elif winner == -1:
			print('winner is -1')
		elif winner == 1:
			print('winner is 1')
		self.reset_game()

	def reset_game(self):
		self.grid = [[0 for _ in range(3)] for _ in range(3)]
		self.__running = True
		self.__player = self.first_player

	def computer_player(self):
		if self.model is not None:
			piece_pos_chosen = self.model.predict(self.grid)
			self.add_piece(piece_pos_chosen)

	def human_player(self, event):
		if event.type == pygame.MOUSEBUTTONUP:
			pos = pygame.mouse.get_pos()
			self.add_piece(pos)

	def add_piece(self, pos):
		if self.mouse_position(pos):
			if self.check_win() != 0:
				#if is computer player
				if self.__player == -1:
					# add_train_data in self.replay_memory
					# reward = 1
					pass
				else:
					# computer loss the game
					# add_train_data in self.replay_memory
					# reward = 1
				winner = self.check_win()
				self.finish_game(winner)
			else:
				#if is computer player
				if self.__player == -1:
					# add_train_data in self.replay_memory
					# reward = 0.1
					pass

				self.__player = -1 * self.__player
		else:
			if self.__player == -1:
				# add_train_data in self.replay_memory && train
				# wrong predition
				# reward = -10
				pass
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
	def mouse_position(self, pos):
		row_height = SCREEN_WIDTH // 3

		col = pos[0] // row_height
		row = pos[1] // row_height

		if self.grid[row][col] == 0:
			self.grid[row][col] = self.__player
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
	game = Tictactoe()
	game.on_execute()
	


if __name__ == '__main__':
	main()