from numpy import exp, array, random, dot
#input 4 * 3
#hidden layer1 3 * 5
#hidden layer2 5 * 4
#hidden layer3 4 * 1
#output 4 * 1
class NeuralNetwork():
	def __init__(self):
		random.seed(1)
		self.weight1 = random.random((3,5))
		self.weight2 = random.random((5,4))
		self.weight3 = random.random((4,1))
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			hidden_layer1 = self.__sigmoid(dot(training_set_inputs, self.weight1))
			hidden_layer2 = self.__sigmoid(dot(hidden_layer1, self.weight2))
			output = self.__sigmoid(dot(hidden_layer2, self.weight3))

			#backward
			del4 = (training_set_outputs - output) * self.__sigmoid_derivative(output)
			del3 = dot(self.weight3, del4.T) * (self.__sigmoid_derivative(hidden_layer2).T)
			del2 = dot(self.weight2, del3) * (self.__sigmoid_derivative(hidden_layer1).T)

			adjustment3 = dot(hidden_layer2.T, del4)
			adjustment2 = dot(hidden_layer1.T, del3.T)
			adjustment1 = dot(training_set_inputs.T, del2.T)

			self.weight1 += adjustment1
			self.weight2 += adjustment2
			self.weight3 += adjustment3


	def think(self, inputs):
		return self.__sigmoid(dot(dot(dot(inputs, self.weight1), self.weight2), self.weight3))



if __name__ == "__main__":

	neural_network = NeuralNetwork()

	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T

	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print("Considering new situation [1, 0, 0] -> ?: ")
	print(neural_network.think(array([1, 0, 0])))