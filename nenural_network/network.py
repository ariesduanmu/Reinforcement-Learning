import numpy as np
from numpy import exp, dot, random, sqrt, log, nan_to_num, zeros, array
class Network():
	def __init__(self, size):
		self.size = size
		self.biases = [random.randn(y,1) for y in self.size[1:]]
		self.weights = [random.randn(y,x)/sqrt(x) for x, y in zip(self.size[:-1], self.size[1:])]
	def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
		
		for e in range(epochs):
			random.shuffle(training_data)
			mini_batchs = [training_data[k:k+mini_batch_size] for k in range(mini_batch_size)]
			for mini_batch in mini_batchs:
				self.update_mini_batch(mini_batch, learning_rate)
			print("Epoch {0}: {1}".format(j, self.evaluate(test_data)))
		
	def update_mini_batch(self, mini_batch, learning_rate):
		nabla_w = [zeros(w.shape) for w in self.weights]
		nabla_b = [zeros(b.shape) for b in self.biases]
		for x, y in mini_batch:
			delta_nabla_w, delta_nabla_b = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w - (learning_rate / len(mini_batch) * nw) for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (learning_rate / len(mini_batch) * nb) for b, nb in zip(self.biases, nabla_b)]

	def sigmoid(self, x):
		return 1.0 / (1.0 + exp(-x))
	def sigmoid_derivative(self, x):
		return self.sigmoid(x) / (1.0 - self.sigmoid(x))
	def feed_forward(self, input_):
		for w, b in zip(self.weights, self.biases):
			input_ = self.sigmoid(dot(w, input_) + b)
		return input_
	def backprop(self, x, y):
		nabla_w = [zeros(w.shape) for w in self.weights]
		nabla_b = [zeros(b.shape) for b in self.biases]

		activations = [x]
		activation = x
		zs = []
		for w, b in zip(self.weights, self.biases):
			z = dot(w, activation) + b
			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)
		delta = self.cost_derivative(activation[-1], y) * self.sigmoid_derivative(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = dot(delta, activations[-2].transpose())

		for l in range(2, len(self.size)):
			z = zs[-l]
			sp = self.sigmoid_derivative(z)
			delta = dot(nabla_w[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = dot(delta, activations[-l-1].transpose())
		return nabla_w, nabla_b
	def cost_derivative(self, prediction, y):
		return -np.sum(nan_to_num(y * log(prediction) + (1 - y)*log(1 - prediction)))
	def evaluate(self, test_data):
		x,ys = test_data
		prediction = self.feed_forward(x)
		return sum(abs(p-y) for p, y in zip(prediction, ys)) / len(ys)
training_data_x = array([[0,1,0],[1,0,1],[1,1,1]])
training_data_y = array([1,0,0])
training_data = [(x,y) for x, y in zip(training_data_x, training_data_y)]
network = Network([3,2,1])
network.train(training_data, 10, 1, 0.01, training_data)
