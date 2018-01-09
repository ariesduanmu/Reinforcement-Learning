from keras.models import Sequential
from keras.layers import Dense, Flatten
from collections import deque
import tensorflow as tf
import numpy as np
import gym

import random
import abc
import os

class QLearningAgent:
    def __init__(self, observation_shape, acition_size):
        self.observation_shape = observation_shape
        self.acition_size = acition_size
        self.epsilon = 0.7
        self.gamma = 0.9
        self.mb_size = 50

        self.model = self.buildmodel()

    @abc.abstractmethod
    def buildmodel(self):
        return None

    def predict(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.acition_size)
        else:
            return np.argmax(self.model.predict(state))
    
    def replay(self, memory):
        if len(memory) < self.mb_size:
            return
        minibatch = random.sample(memory, self.mb_size)
        inputs_shape = (self.mb_size,) + memory[0][0].shape[1:]
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((self.mb_size, self.acition_size))

        for i in range(self.mb_size):
            state, action, reward, new_state, done = minibatch[i]
            inputs[i:i+1] = np.expand_dims(state, axis=0)
            targets[i] = self.model.predict(state)
            Q_sa = self.model.predict(new_state)

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(Q_sa)

        self.model.train_on_batch(inputs, targets)



class DeepQLearningAgent(QLearningAgent):
    def buildmodel(self):
        model = Sequential()
        model.add(Dense(20, input_shape=(2,) + self.observation_shape, kernel_initializer='uniform', activation='relu'))
        model.add(Flatten())
        model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(self.acition_size, kernel_initializer='uniform', activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        if os.path.isfile('mountaincar-v1.h5'):
            model.load_weights('mountaincar-v1.h5')
        return model
    def close_save(self):
        self.model.save_weights('mountaincar-v1.h5', overwrite=True)



class GymRunner:
    def __init__(self, env_name, iter_max):
        self.env = gym.make(env_name)
        self.iter_max = iter_max
        self.memory = deque()

    @property
    def observation_shape(self):
        return self.env.observation_space.shape

    @property
    def action_size(self):
        return self.env.action_space.n 

    def train(self, agent):
        obs = self.env.reset()
        state = self.obs_to_state(obs)
        for i in range(self.iter_max):
            action = agent.predict(state)
            new_obs, reward, done, _ = self.env.step(action)
            new_state = self.obs_to_state(new_obs)
            self.memory.append((state, action, reward, new_state, done))
            state = new_state
            if done:
                obs = self.env.reset()
                state = self.obs_to_state(obs)
        agent.replay(self.memory)

    def test(self, agent):
        obs = self.env.reset()
        state = self.obs_to_state(obs)
        total_reward = 0
        for _ in range(10000):
            self.env.render()
            action = agent.predict(state)
            obs, reward, done, _ = self.env.step(action)
            state = self.obs_to_state(obs)
            total_reward += reward
        print("total reward:{}".foramt(total_reward))


    def obs_to_state(self, obs):
        obs = np.expand_dims(obs, axis = 0)
        return np.stack((obs,obs), axis = 1)

if __name__ == "__main__":
    game = GymRunner("MountainCar-v0", 10000)
    agent = DeepQLearningAgent(game.observation_shape, game.action_size)
    #game.train(agent)
    #agent.close_save()
    game.test(agent)


        