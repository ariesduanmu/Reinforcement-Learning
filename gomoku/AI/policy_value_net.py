import theano
import theano.tensor as T
import lasagne

class PolicyValueNet():
    def __init__(self, board_width, board_height, net_params = None):
        self.board_width = board_width
        self.board_height = board_height

    def create_policy_value_net(self):
        self.state_input = T.tensor4('state')
        self.winner = T.vector('winner')
        self.mcts_probs = T.matrix('mcts_probs')
        network = lasagne.layers.InputLayer(shape = (None, 4, self.board_width, self.board_height),
                                            input_var = self.state_input)

        network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), pad="same")
        network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3, 3), pad='same')

        policy_net = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=(1, 1))
        self.policy_net = lasagne.layers.DenseLayer(policy_net, num_units=self.board_width*self.board_height, 
                                            nonlinearity=lasagne.nonlinearities.softmax)
        value_net = lasagne.layers.Conv2DLayer(network, num_filters=2, filter_size=(1, 1))
        value_net = lasagne.layers.DenseLayer(value_net, num_units=64)
        self.value_net = lasagne.layers.DenseLayer(value_net, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

        self.action_probs, self.value = lasagne.layers.get_output([self.policy_net, self.value_net])
        self.policy_value = theano.function([self.state_input], [self.action_probs, self.value] ,allow_input_downcast=True)

    def policy_value_fn(self, board):
        pass

    def _loss_train_op(self):
        pass

    def get_policy_param(self):
        pass
