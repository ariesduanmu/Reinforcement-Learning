import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.optimizers import Adam

import os.path


gamma = 0.97

class PongModel(object):
	def __init__(self, file_name):
		self.file_name = file_name
		self.model = self.build_model()
		

	def build_model(self):
		model = Sequential()
		model.add(Convolution2D(32, 5, 5, subsample=(1,1), border_mode='same', input_shape=(1,80,80)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 5, 5, subsample=(1,1), border_mode='same'))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same'))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dense(2))

		adam = Adam(lr=1e-6)
		model.compile(loss='mse', optimizer=adam)

		if os.path.exists(self.file_name):
			model.load_weights(self.file_name)
		return model
	
	def train_network(self, replay_memory):
		total_loss_cur = 0
		for ix in range(len(replay_memory) - 1):# why ignore the last one
			state_t = replay_memory[ix][0]
			state_t_r = replay_memory[ix][2]
			action_t = replay_memory[ix][1]
			reward_t = replay_memory[ix][3]

			target_t = self.model.predict(state_t, 1)[0]
			Q_pred_t1 = self.model.predict(state_t_r, 1)[0]

			if action_t[0] == 1:
				target_t[0] = reward_t + gamma * np.amax(Q_pred_t1)
			else:
				target_t[1] = reward_t + gamma * np.amax(Q_pred_t1)
			print(target_t)
			hist = self.model.fit(state_t, np.atleast_2d(target_t), batch_size=1, epochs=1)
			total_loss_cur += hist.history['loss'][0]
		print(total_loss_cur/len(replay_memory))
		print("Done updating weights!")
		self.model.save_weights(self.file_name)
