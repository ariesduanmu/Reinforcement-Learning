import numpy as np
import random
import abc
from collections import deque


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.batch_size = 32

        self.model = self.build_model()
        self.memory = deque(maxlen = 2000)

    @abc.abstractmethod
    def build_model(self):
        return None

    def select_action(self, state, do_train=True):
        if do_train and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        return np.argmax(self.model.predict(state)[0])

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.max(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs = 1, verbose = 0)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay