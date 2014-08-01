__author__ = 'davidnola'

import glob
import cPickle
import scipy
import scipy.io


class UniversalSegment:
    def __init__(self):
        pass


SUBJECTS = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']


def preprocess_subject(subject):
    files = glob.glob('/Users/davidnola/Machine_Learning_Research/Data/'+ subject+'/*.mat')
    f_iter = iter(files)



    #test = cPickle.load(open(subject+'_TEST.pkl', 'rb'))
    #train = cPickle.load(open(subject+'.pkl', 'rb'))


    segments = []

    for f in files:
        s = UniversalSegment()
        matfile = scipy.io.loadmat(f)
        s.data = matfile['data']
        s.frequency = matfile['freq']
        s.name = f[f.rindex('/')+1:]
        s.features={}

        print f
        if 'test' in f:
            #s.calculate_features()
            s.data = []
            test_data.append(s)
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
        clips.append(s)
    print "wait ", subject
    #cPickle.dump(train, open(subject+'.pkl', 'wb'))
    #cPickle.dump(test, open(subject+'_TEST.pkl', 'wb'))
    print "done ", subject