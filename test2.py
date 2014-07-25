__author__ = 'davidnola'
__author__ = 'davidnola'
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import cPickle
import StackedDenoisingAutoencoder

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

    s = StackedDenoisingAutoencoder.SDAManager()
    s.fit(features, classes)
    print s.predict(features_test)

test_model()