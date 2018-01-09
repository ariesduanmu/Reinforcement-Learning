import numpy as np
import pickle
import gym
import os

H = 200
D = 80 * 80

def initial_model():
    if os.path.isfile("model.p"):
        return pickle.load(open("model.p", "rb"))
    
    return {
      "W1" : np.random.randn(H, D) / np.sqrt(D),
      "W2" : np.random.randn(H) / np.sqrt(H)
    }

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward_feed(model, x):
    h = np.dot(model["W1"], x)
    h[h<0] = 0
    logp = np.dot(model["W2"], h)
    p = sigmoid(logp)
    return p, h

def train(episodes = 1000, render = False):
    env = gym.make("Pong-v0")
    observation = env.reset()
    model = initial_model()

    for episode in range(episodes):
        if render: env.render()
        print(env.action_space.sample())
        observation, reward, done, info = env.step(env.action_space.sample())

if __name__ == "__main__":
    train()