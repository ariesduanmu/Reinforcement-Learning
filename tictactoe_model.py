import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.optimizers import Adam

import os.path

class TttModel(object):
	def __init__(self, filename):
		self.filename = filename
		self.model = self.build_model()
	def build_model(self):
		model = Sequential()
		#[3*3] -> [9]
		model.compile(loss='mse', optimizer=adam)

		if os.path.exists(self.filename):
			model.load_weights(self.filename)
		return model
	def train_network(self):
		#loss -1
		#win +1
		#draw 0
		#wrong piece -10
		
