from __future__ import absolute_import, division, print_function

import pygame
import random
import sys
from pygame.locals import *
import numpy as np
import os.path
from pong_model import PongModel
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 300

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

BALL_RADIUS = 5

PADDLE_SPEED = 4
BALL_SPEED = 5
MAX_SPEED = 6

BLACK = (0,0,0)
WHITE = (255,255,255)

MAX_SIZE = 5000

FILE_NAME = 'model_conv.keras'
class Pong(object):
	def __init__(self, model = None):

		self.reset_game()
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT),pygame.HWSURFACE | pygame.DOUBLEBUF)
		pygame.display.set_caption('Pong!')
		

		self.action_chosen = np.asarray([0,0])
		self.model = model

		self.final_img = np.zeros((80,80))

		self.reward_cur = 0.05

		self.replay_memory = []

		self.__running = True
		self.epsilon = 0.25


	def on_excute(self):
		_= self.retrieve_image()
		while self.__running:
			if len(self.replay_memory) < MAX_SIZE:
				self.game_board_init()
				self.ball_move()
				for event in pygame.event.get():
					self.on_event(event)
				self.computer_move()
				self.on_render()
			else:
				if self.model is not None:
					self.model.train_network(self.replay_memory)
					self.epsilon *= 0.97
				self.replay_memory = []
	def on_event(self, event):
		if event.type == pygame.QUIT:
			self.__running = False
		# human_play
		#self.human_move(event)
	def on_render(self):
		self.render_ball()
		self.render_paddle()
		pygame.display.update()
	def reset_game(self):
		self.player_pos = [0, SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2]
		self.ball_pos = [PADDLE_WIDTH + BALL_RADIUS, SCREEN_HEIGHT // 2]
		self.initialize_ball_velocity()
	def game_board_init(self):
		self.screen.fill(BLACK)
	def initialize_ball_velocity(self):
		self.ball_vec = [int(round(random.uniform(0.7,1.1) * BALL_SPEED)), int(round(random.uniform(0.7,1.1) * BALL_SPEED))]
		if random.uniform(0,1) > 0.5:
			self.ball_vec[1] *= -1

	def render_ball(self):
		pygame.draw.circle(self.screen, WHITE, self.ball_pos, BALL_RADIUS)
	def render_paddle(self):
		pygame.draw.rect(self.screen, WHITE, (self.player_pos[0], self.player_pos[1], PADDLE_WIDTH, PADDLE_HEIGHT))
	def ball_move(self):
		self.ball_pos[0] += self.ball_vec[0]
		self.ball_pos[1] += self.ball_vec[1]
		self.reward_cur = 0.05
		#outside
		if self.ball_pos[0] < BALL_RADIUS + PADDLE_WIDTH//2:
			self.reward_cur = -1
			if self.model is not None:
				self.model.train_network(self.replay_memory)
				self.epsilon *= 0.97
			self.replay_memory = []
			self.reset_game()
		
		#bounder change velocity direction
		if self.ball_pos[1] > SCREEN_HEIGHT - BALL_RADIUS or self.ball_pos[1] < BALL_RADIUS:
			if self.ball_pos[1] < BALL_RADIUS:
				self.ball_pos[1] = BALL_RADIUS
			else:
				self.ball_pos[1] = SCREEN_HEIGHT - BALL_RADIUS
			self.ball_vec[1] *= -1
		if self.ball_pos[0] > SCREEN_WIDTH - BALL_RADIUS:
			self.ball_pos[0] = SCREEN_WIDTH - BALL_RADIUS
			self.ball_vec[0] *= -1

	    #collision
		if self.check_collision():
			print('Collided!')
			self.reward_cur = 1
			gap = abs(self.ball_pos[1] - (self.player_pos[1] + PADDLE_HEIGHT/2))*1.0 / (PADDLE_HEIGHT / 2)
			self.ball_vec[0] = int(round(self.ball_vec[0] * (-1) * (max(1, gap + 0.3))))
			self.ball_vec[1] = int(round(self.ball_vec[1] * (max(0.9, gap + 0.4))))
			self.ball_pos[0] += int(round(self.ball_vec[0]))
		self.ball_vec[0] = max(-MAX_SPEED, min(MAX_SPEED, self.ball_vec[0]))
		self.ball_vec[1] = max(-MAX_SPEED, min(MAX_SPEED, self.ball_vec[1]))

		self.ball_pos[0] = int(round(self.ball_pos[0]))
		self.ball_pos[1] = int(round(self.ball_pos[1]))
	    

	def paddle_move(self):
		if self.action_chosen[0] == 1:
			self.player_pos[1] -= PADDLE_SPEED
		elif self.action_chosen[1] == 1:
			self.player_pos[1] += PADDLE_SPEED

		if self.player_pos[1] < 0:
			self.player_pos[1] = 0
		elif self.player_pos[1] > SCREEN_HEIGHT - PADDLE_HEIGHT:
			self.player_pos[1] = SCREEN_HEIGHT - PADDLE_HEIGHT
	def retrieve_image(self):
		old_img = self.final_img

		img = pygame.surfarray.array3d(pygame.transform.scale(pygame.display.get_surface(), (80, 80)))
		avgs = [[(r * 0.33 + g * 0.33 + b * 0.33) for (r, g, b) in col] for col in img]
		self.final_img = np.array([[avg for avg in col] for col in avgs])

		return self.final_img - old_img
	def computer_move(self):
		current_state = self.retrieve_image()
		current_state = np.expand_dims(current_state, axis=0)
		current_state = np.expand_dims(current_state, axis=0)
		self.action_chosen = np.asarray([0, 0])
		if self.model is not None:
			actions_possible = [0,0]
			if not os.path.exists(FILE_NAME):
				action_idx = random.randrange(2)
				actions_possible[action_idx] = 1
				actions_possible[1-action_idx] = 0
			else:
				actions_possible = self.model.model.predict(current_state, 1)[0]
				if random.uniform(0, 1) < self.epsilon:
					action_idx = random.randrange(2)
					actions_possible[action_idx] = 1
					actions_possible[1-action_idx] = 0

			if actions_possible[0] > actions_possible[1]:
				self.action_chosen = np.asarray([1, 0])
			else:
				self.action_chosen = np.asarray([0, 1])
			if len(self.replay_memory) > 0:
				self.replay_memory[-1][2] = current_state
				self.replay_memory[-1][3] = self.reward_cur
			reward_dummy = 0
			next_state_dummy = current_state

			self.replay_memory.append([current_state, self.action_chosen, next_state_dummy, reward_dummy])
		self.paddle_move()
	def human_move(self, event):
		if event.type == pygame.MOUSEBUTTONUP:
			pos = pygame.mouse.get_pos()
			if pos[1] < self.player_pos[1] + PADDLE_HEIGHT // 2:
				self.action_chosen = np.asarray([1 ,0])
			elif pos[1] > self.player_pos[1] + PADDLE_HEIGHT // 2:
				self.action_chosen = np.asarray([0 ,1])
			self.paddle_move()

	def check_collision(self):
		rect1 = Rect(self.player_pos[0], self.player_pos[1], PADDLE_WIDTH, PADDLE_HEIGHT)
		rect2 = Rect(self.ball_pos[0] - BALL_RADIUS, self.ball_pos[1] - BALL_RADIUS, 2 * BALL_RADIUS, 2 * BALL_RADIUS)
		if rect2.colliderect(rect1):
			return True
		return False

model = PongModel(file_name = FILE_NAME)
pong = Pong(model=model)
pong.on_excute()