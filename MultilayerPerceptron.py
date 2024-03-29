__author__ = 'davidnola'

import theano
import theano.tensor as T
import numpy as np

print theano
import cPickle
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




    def output(self, x):

        for layer in self.layers:
            x = layer.output(x)
        #print "out"
        return x

    def squared_error(self, x, y):

        p_1 = self.output(x)
        xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
        cost = xent.mean()# + 0.01 * (w ** 2).sum()

        #return cost
        return T.sum((self.output(x) - y)**2)




class MultilayerPerceptronManager:
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


    def fit(self, X, y):
        features = []
        for i in range(len(X[0])):
            toadd = []
            for x in X:
                toadd.append(x[i])
            features.append(toadd)
            #print toadd

        X = np.vstack(features).astype(theano.config.floatX)

        y = np.array(y).astype(theano.config.floatX)
        self.layer_sizes = [X.shape[0], 100, 1]
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

        self.mlp_input = T.matrix('mlp_input')

        self.mlp_target = T.vector('mlp_target')

        self.learning_rate = 0.00001
        self.momentum = .9

        self.cost = self.mlp.squared_error(self.mlp_input, self.mlp_target)

        self.train = theano.function([self.mlp_input, self.mlp_target], self.cost,
                                updates=self.gradient_updates_momentum(self.cost, self.mlp.params, self.learning_rate, self.momentum))

        self.mlp_output = theano.function([self.mlp_input], self.mlp.output(self.mlp_input),
                                        #mode=theano.compile.MonitorMode(    pre_func=inspect_inputs,)
                        )


        self.iteration = 0

        self.max_iteration = 20000
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



class EEGSegment:
    features = {
            # 'channel_variances': [],
            # 'channel_1sig_times_exceeded': [],
            # 'channel_2sig_times_exceeded': [],
            # 'channel_3sig_times_exceeded': [],
        }

    def __init__(self):
        self.name = ""
        self.data = []
        self.latency = -1
        self.seizure = False
        self.frequency = 0
        self.features = {
            # 'channel_variances': [],
            # 'channel_1sig_times_exceeded': [],
            # 'channel_2sig_times_exceeded': [],
            # 'channel_3sig_times_exceeded': [],
        }



    def segment_info(self, showdata=False):
        print "SEIZURE: " + str(self.seizure)
        print "LATENCY: " + str(self.latency)

        if showdata:
            print "DATA: "
            for l in self.data:
                print l
                print

        print "END\n"

    def calculate_features(self):


        self.features['channel_variances'] = []
        iter = 0
        for d in self.data:
            x = np.var(d)
            self.features['channel_variances'].append(x)


        cursig = 0
        for siglevel in ['channel_1sig_times_exceeded', 'channel_2sig_times_exceeded', 'channel_3sig_times_exceeded']:
            self.features[iter] = []
            cursig+=1
            iter = 0.0
            for d in self.data:
                iter += 1.0
                stddev = np.std(d)
                mean = np.mean(d)
                prior = d[0]
                exceeded = 0
                for x in d[1:]:
                    if prior < (cursig*stddev + mean) and x > (cursig*stddev + mean):
                        exceeded+=1
                    prior = x
                self.features[siglevel].append(exceeded)

        self.data = []
        print self.features



