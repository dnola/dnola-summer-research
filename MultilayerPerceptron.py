__author__ = 'davidnola'

import theano
import theano.tensor as T
import numpy as np

print theano


theano.config.optimizer = 'None'

class Storage:
    model = None


class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.

        :parameters:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_output, n_input = W_init.shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output,)
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output,
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               # The name parameter is solely for printing purporses
                               name='W',
                               # Setting borrow=True allows Theano to use user memory for this object.
                               # It can make code slightly faster by avoiding a deep copy on construction.
                               # For more details, see
                               # http://deeplearning.net/software/theano/tutorial/aliasing.html
                               borrow=True)
        # We can force our bias vector b to be a column vector using numpy's reshape method.
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting (described below)
        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               # Theano allows for broadcasting, similar to numpy.
                               # However, you need to explicitly denote which axes can be broadcasted.
                               # By setting broadcastable=(False, True), we are denoting that b
                               # can be broadcast (copied) along its second dimension in order to be
                               # added to another variable.  For more information, see
                               # http://deeplearning.net/software/theano/library/tensor/basic.html
                               broadcastable=(False, True))
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]


    def output(self, x):
        '''
        Compute this layer's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation is None else self.activation(lin_output))


class MLP(object):
    def __init__(self, W_init, b_init, activations):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)

        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params




    def output(self, x):
        '''
        Compute the MLP's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        #return T.sum((self.output(x) - y)**2)
        #return -T.mean(T.log(self.output(x))[T.arange(y.shape[0]), y])

        p_1 = self.output(x)
        xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
        cost = xent.mean()# + 0.01 * (w ** 2).sum()

        return cost




class MultilayerPerceptronManager:
    def __init__(self):
        pass
        # self.W_init = []
        # self.b_init = []
        # self.activations = []
        # self.learning_rate = 0.01
        # self.momentum = 0.9
        # self.mlp_input = T.matrix('mlp_input')
        # # ... and the desired output
        # self.mlp_target = T.vector('mlp_target')
        # self.mlp = MLP(self.W_init, self.b_init, self.activations)
        # self.cost = self.mlp.squared_error(self.mlp_input, self.mlp_target)
        # # Create a theano function for training the network
        # self.train = theano.function([self.mlp_input, self.mlp_target], self.cost,
        #                         updates=self.gradient_updates_momentum(self.cost, self.mlp.params, self.learning_rate, self.momentum))
        # # Create a theano function for computing the MLP's output given some input
        # self.mlp_output = theano.function([self.mlp_input], self.mlp.output(self.mlp_input))


    def gradient_updates_momentum(self, cost, params, learning_rate, momentum):
        '''
        Compute updates for gradient descent with momentum

        :parameters:
            - cost : theano.tensor.var.TensorVariable
                Theano cost function to minimize
            - params : list of theano.tensor.var.TensorVariable
                Parameters to compute gradient against
            - learning_rate : float
                Gradient descent learning rate
            - momentum : float
                Momentum parameter, should at least 0 (standard gradient descent) and less than 1

        :returns:
            updates : list
                List of updates, one for each parameter
        '''
        # Make sure momenum is a sane value
        assert momentum < 1 and momentum >= 0
        # List of update steps for each parameter
        updates = []
        # Just gradient descent on cost
        for param in params:
            # For each parameter, we'll create a param_update shared variable.
            # This variable will keep track of the parameter's update step across iterations.
            # We initialize it to 0
            self.param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            # Each parameter is updated by taking a step in the direction of the gradient.
            # However, we also "mix in" the previous step according to the given momentum value.
            # Note that when updating param_update, we are using its old value and also the new gradient step.
            updates.append((param, param - learning_rate*self.param_update))
            # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
            updates.append((self.param_update, momentum*self.param_update + (1. - momentum)*T.grad(cost, param)))
        return updates


    def fit(self, X, y):
        features = []
        for i in range(len(X[0])):
            toadd = []
            for x in X:
                toadd.append(x[i])
            features.append(toadd)
            #print toadd

        X = np.vstack(features)

        y = np.array(y)
        self.layer_sizes = [X.shape[0], 5, 1]
        print "Layer Sizes ", self.layer_sizes


        # Set initial parameter values
        self.W_init = []
        self.b_init = []
        self.activations = []
        for n_input, n_output in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # Getting the correct initialization matters a lot for non-toy problems.
            # However, here we can just use the following initialization with success:
            # Normally distribute initial weights
            self.W_init.append(np.random.randn(n_output, n_input))
            # Set initial biases to 1
            self.b_init.append(np.ones(n_output))
            # We'll use sigmoid activation for all layers
            # Note that this doesn't make a ton of sense when using squared distance
            # because the sigmoid function is bounded on [0, 1].
            self.activations.append(T.nnet.sigmoid)
        # Create an instance of the MLP class
        self.mlp = MLP(self.W_init, self.b_init, self.activations)

        # Create Theano variables for the MLP input
        self.mlp_input = T.matrix('mlp_input')
        # ... and the desired output
        self.mlp_target = T.vector('mlp_target')
        # Learning rate and momentum hyperparameter values
        # Again, for non-toy problems these values can make a big difference
        # as to whether the network (quickly) converges on a good local minimum.
        self.learning_rate = 0.01
        self.momentum = 0.9
        # Create a function for computing the cost of the network given an input
        self.cost = self.mlp.squared_error(self.mlp_input, self.mlp_target)
        # Create a theano function for training the network
        self.train = theano.function([self.mlp_input, self.mlp_target], self.cost,
                                updates=self.gradient_updates_momentum(self.cost, self.mlp.params, self.learning_rate, self.momentum))
        # Create a theano function for computing the MLP's output given some input
        self.mlp_output = theano.function([self.mlp_input], self.mlp.output(self.mlp_input))


        # Keep track of the number of training iterations performed
        self.iteration = 0
        # We'll only train the network with 20 iterations.
        # A more common technique is to use a hold-out validation set.
        # When the validation error starts to increase, the network is overfitting,
        # so we stop training the net.  This is called "early stopping", which we won't do here.
        self.max_iteration = 10000
        while self.iteration < self.max_iteration:
            #print "iteration: ", self.iteration
            # Train the network using the entire training set.
            # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
            # where only a subset (or a single point) of the training set is used at each iteration.
            # This can also help the network to avoid local minima.
            #print "train"
            self.current_cost = self.train(X, y)
            # Get the current network output for all points in the training set
            #print "output"
            self.current_output = self.mlp_output(X)
            # We can compute the accuracy by thresholding the output
            # and computing the proportion of points whose class match the ground truth class.
            #print "acc"
            accuracy = np.mean((self.current_output > .5) == y)
            # Plot network output after this iteration
            self.iteration += 1

        self.model = self.mlp

    def predict(self, X):
        features = []
        for i in range(len(X[0])):
            toadd = []
            for x in X:
                toadd.append(x[i])
            features.append(toadd)

        X = np.vstack(features)
        self.mlp = self.model
        self.mlp_input = T.matrix('mlp_input')
        self.mlp_output = theano.function([self.mlp_input], self.mlp.output(self.mlp_input))
        ret =  self.mlp_output(X)
        ret = [round(x, 7) for x in ret[0]]
        #print "MLP: "

        #print ret

        return ret

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.array(y)

        print "pred", predictions
        print "act", y
        classes =  [ (x > .5) * 1.0 for x in self.predict(X)]
        print "test", classes

        accuracy = np.mean(classes == y)
        print "accuracy: ", accuracy
        return .6

