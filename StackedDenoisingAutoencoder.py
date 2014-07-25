__author__ = 'davidnola'

import theano
import theano.tensor as T
import numpy as np
import cPickle
import DenoisingAutoencoder as DA

print theano
from theano.tensor.shared_randomstreams import RandomStreams
theano.config.optimizer = 'None'

class Storage:
    model = None

class Layer(object):
    def __init__(self, W_init, b_init, activation):

        n_output, n_input = W_init.shape

        # Turn lists of W_init and b_init into proper shared Theano variables

        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               name='W',
                               )

        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name='b',
                               broadcastable=(False, True))

        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        return self.activation(lin_output)


class MLP(object):
    def __init__(self, W_init, b_init, activations):

        self.layers = []

        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))


        self.params = []
        for layer in self.layers:
            self.params += layer.params




    def output(self, x, down = 0):

        if down != 0:
            active = self.layers[:-down]
        else:
            active = self.layers

        for layer in active:
            x = layer.output(x)
        #print "out"
        return x

    def squared_error(self, x, y):

        p_1 = self.output(x)
        xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
        cost = xent.mean()# + 0.01 * (w ** 2).sum()

        #return cost
        return T.sum((self.output(x) - y)**2)




class SDAManager:
    def __init__(self):
        pass

    def gradient_updates_momentum(self, cost, params, learning_rate, momentum):

        assert momentum < 1 and momentum >= 0

        updates = []

        for param in params:

            self.param_update = theano.shared( param.get_value()*0.5, broadcastable=param.broadcastable)

            updates.append((param, param - learning_rate*self.param_update))
            updates.append((self.param_update, momentum*self.param_update + (1. - momentum)*T.grad(cost, param)))
        #print "here"
        return updates

    def generate_pretraining_functions(self):
        corruption_level = T.scalar('corruption')  # amount of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use


        pretrain_fns = []
        for dA in self.autoencoders:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level, learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                    outputs=cost,
                    updates=updates,
                    givens={self.mlp_input: self.X},
                    mode='DebugMode')
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def fit(self, X, y):
        features = []
        for i in range(len(X[0])):
            toadd = []
            for x in X:
                toadd.append(x[i])
            features.append(toadd)
            #print toadd

        X = np.vstack(features)
        self.X = X

        y = np.array(y)
        self.layer_sizes = [X.shape[0], 30, 4, 1]
        print "Layer Sizes ", self.layer_sizes


        self.W_init = []
        self.b_init = []
        self.activations = []
        for n_input, n_output in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):

            self.W_init.append(np.random.randn(n_output, n_input))

            self.b_init.append(np.ones(n_output))

            self.activations.append(T.nnet.sigmoid)
            #self.activations.append(T.tanh)

        self.mlp = MLP(self.W_init, self.b_init, self.activations)


        self.autoencoders = []
        print "new"



        self.mlp_input = T.matrix('mlp_input')
        self.mlp_target = T.vector('mlp_target')



        inputs = []
        inputs.append(self.mlp_input)

        for i in reversed(range(len(self.mlp.layers)-1)):
            inputs.append(self.mlp.output(self.mlp_input, i))

        in_iter = iter(inputs)
        in_len_iter = iter(self.layer_sizes)
        out_len_iter = iter(self.layer_sizes[1:])
        for layer in self.mlp.layers:
            print layer.W.get_value()
            next = in_iter.next()
            a = DA.dA(input = next, n_visible = in_len_iter.next(), n_hidden = out_len_iter.next(), W=layer.W, bhid=layer.b)
            self.autoencoders.append(a)




        self.learning_rate = 0.00001
        self.momentum = .9

        self.cost = self.mlp.squared_error(self.mlp_input, self.mlp_target)

        self.train = theano.function([self.mlp_input, self.mlp_target], self.cost,
                                updates=self.gradient_updates_momentum(self.cost, self.mlp.params, self.learning_rate, self.momentum))

        self.mlp_output = theano.function([self.mlp_input], self.mlp.output(self.mlp_input),
                                        #mode=theano.compile.MonitorMode(    pre_func=inspect_inputs,)
                        )




        print "more new"

        self.pretrain_functions = self.generate_pretraining_functions()
        #
        # for xi in range(len(self.pretrain_functions)):
        #     self.pretrain_functions[xi](corruption=0.2, lr=0.00001 )



        rng = np.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        da = DA.dA(theano_rng=theano_rng, input=self.mlp_input,
                n_visible=16, n_hidden=500)

        cost, updates = da.get_cost_updates(corruption_level=0.,
                                            learning_rate=self.learning_rate)

        train_da = theano.function([], cost, updates=updates,
             givens={self.mlp_input: X})

        for xz in range(100):
            train_da()


        self.iteration = 0

        self.max_iteration = 1000
        while self.iteration < self.max_iteration:
            if self.iteration%50 == 0:
                print self.iteration

            self.current_cost = self.train(X, y)

            #print "done with train"

            self.current_output = self.mlp_output(X)
            #print self.current_output[0]

            accuracy = np.mean((self.current_output > .5) == y)
            #print "acc", accuracy

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

        print "pred\t", predictions
        print "act\t", list(y)
        classes =  [ (x > .5) * 1.0 for x in self.predict(X)]
        print "test", classes

        score = 0
        for i in range(len(y)):
            if y[i] == classes[i]:
                score+=1.0

        #accuracy = np.mean( [c == yiter.next() for c in classes])
        accuracy = score/float(len(y))

        print "accuracy: ", accuracy
        return .6




def test_model():
    test = cPickle.load(open("SummerResearchData/"+'Dog_1'+'_TEST.pkl', 'rb'))
    train = cPickle.load(open("SummerResearchData/"+'Dog_1'+'.pkl', 'rb'))

    features = []
    classes = []

    for c in train:
        #print c.features
        features.append(c.features['channel_variances'])
        classes.append(c.seizure)

    features_test = []
    for c in test:
        #print c.features
        features_test.append(c.features['channel_variances'])

    s = SDAManager()
    s.fit(features, classes)
    print s.predict(features_test)

#test_model()