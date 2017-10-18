import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.optimizers import Adam

import os.path
from nenural_network.first_nenural import NeuralNetwork
from numpy import array, argmax

#loss the game all choose in this game * -1 and human player's choice +0.1
#if win oppsite above
#wrong piece: the last piece reward -10
class TttModel(object):
	def __init__(self, filename):
		self.filename = filename
		#self.model = self.build_model()

		if os.path.exists(filename):
			hidden_weights = self.weight_data_2_matrix(open(filename).read())
			self.model = NeuralNetwork(weight1 = hidden_weights[0], weight2 = hidden_weights[1], weight3 = hidden_weights[2])
		else:
			self.model = NeuralNetwork()
	def build_model(self):
		model = Sequential()
		#[9] -> [9]
		model.compile(loss='mse', optimizer=adam)

		if os.path.exists(self.filename):
			model.load_weights(self.filename)
		return model
	
	def train_network(self, replay_memory):
		#loss -1
		#win +1
		#draw 0.1
		#wrong piece -10
		for ix in range(len(replay_memory)):
			input_data = replay_memory[ix][0]
			choice = replay_memory[ix][1]
			self.model.train(input_data, choice, 10000)
		#save weights
		data = self.matrix_2_weight_data([self.model.weight1, self.model.weight2, self.model.weight3])
		with open(self.filename, 'w+') as f:
			f.write(data)
	def predict(self, input_data):
		choice_pre = self.model.think(input_data)
		choice = argmax(choice_pre)
		print(choice)
		row = choice // 3
		col = choice % 3
		return [row, col]



	def weight_data_2_matrix(self,data):
		hidden_weights = []
		weights = data.split('\n')
		for weight in weights:
			ws = weight.split()
			wm = []
			for w in ws:
				wm.append(list(map(float,w.split(','))))
			hidden_weights += [wm]
		return hidden_weights
	def matrix_2_weight_data(self,weights):
		data = ''
		for weight in weights:
			for i in range(len(weight)):
				data += ','.join(list(map(str,weight[i]))) + ' '
			data += '\n'
		return data

	