__author__ = 'davidnola'

import glob
import cPickle
import scipy
import scipy.io
from math import ceil

class UniversalSegment:
    def __init__(self):
        pass


SUBJECTS = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']


def preprocess_subject(subject):
    files = glob.glob('/Users/davidnola/Machine_Learning_Research/Data/'+ subject+'/*.mat')
    f_iter = iter(files)



    #test = cPickle.load(open(subject+'_TEST.pkl', 'rb'))
    #train = cPickle.load(open(subject+'.pkl', 'rb'))
    target_data = []
    train_data = []


    segments = []

    for f in files:

        matfile = scipy.io.loadmat(f)

        for i in range(4):
            s = UniversalSegment()
            data_cp = matfile['data'][:]
            for d in data_cp:
                size = ceil(d/4.0)
                d = d[ i*size : (i+1)*size]
                print len(data_cp[0]), len(matfile['data'][0])
            s.data = data_cp
            s.frequency = matfile['freq']
            s.name = f[f.rindex('/')+1:]
            s.features={}

            print f
            if 'test' in f:
                #s.calculate_features()
                s.data = []
                target_data.append(s)
                continue

            if 'inter' in f:
                s.seizure = False
                s.latency = -1
            else:
                s.seizure = True
                s.latency = matfile['latency']
                #print s.latency

            #s.segment_info()

            #s.calculate_features()
            s.data = []
            train_data.append(s)

    print "wait ", subject
    #cPickle.dump(train, open(subject+'.pkl', 'wb'))
    #cPickle.dump(test, open(subject+'_TEST.pkl', 'wb'))
    print "done ", subject