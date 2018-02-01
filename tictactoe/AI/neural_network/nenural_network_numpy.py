import numpy as np

def feed_forward(weights, X):
    h2 = weights["W1"] @ X
    a2 = sigmoid(h2)
    h3 = weights["W2"] @ h2
    a3 = sigmoid(h3)

    return h2, a2, h3, a3

def update_weights(weights, player, winner, board_memory, move_memory, learning_rate):

    X, y = process_training_data(player, winner, board_memory, move_memory)

    _, n = y.shape

    h2, a2, h3, a3 = feed_forward(weights, X)

    delta3 = a3 - y
    delta2 = weights["W2"].T @ delta3 * sigmoid_prime(h2)

    W1_grad = (1 / n) * delta2 @ X.T
    W2_grad = (1 / n) * delta3 @ h2.T

    weights["W1"] -= (learning_rate / n) * W1_grad
    weights["W2"] -= (learning_rate / n) * W2_grad

    return weights

def process_training_data(player, winner, board_memory, move_memory):

    X = np.array(board_memory).T
    if winner == player:
        moves = [[1 if i == m else 0 for i in range(9)] for m in move_memory]
    elif winner == 0:
        moves = [[0.5 if i == m else 0 for i in range(9)] for m in move_memory]
    else:
        moves = [[0 for _ in range(9)] for _ in move_memory]

    y = np.array(moves).T

    return X, y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))