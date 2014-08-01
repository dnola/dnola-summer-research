__author__ = 'davidnola'

import glob
import cPickle
import scipy
import scipy.io
from math import ceil
from scipy.ndimage.interpolation import zoom

class UniversalSegment:
    def __init__(self):
        pass


SUBJECTS = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']


one_hots = {}
it = iter(range(12))
for s in SUBJECTS:
    next = it.next()
    one_hots[s] = [1  if x == next else 0 for x in range(12)]

print one_hots

def preprocess_subject(subject):
    files = glob.glob('/Users/davidnola/Data/'+ subject+'/*.mat')
    f_iter = iter(files)



    #test = cPickle.load(open(subject+'_TEST.pkl', 'rb'))
    #train = cPickle.load(open(subject+'.pkl', 'rb'))
    target_data = []
    train_data = []


    segments = []

    for f in files:
        print f
        matfile = scipy.io.loadmat(f)

        for i in range(4):

            s = UniversalSegment()
            data_cp = matfile['data'][:]
            data = []
            for d in range(len(data_cp)):
                size = ceil(len(data_cp[d])/4.0)
                data.append(data_cp[d][ i*size : (i+1)*size])

            #print len(data[0]), len(matfile['data'][0])

            #print data




            s.data = data
            s.frequency = matfile['freq']
            s.name = f[f.rindex('/')+1:]
            s.features={}

            s.features['downsampled_data'] = []

            scale = 100/float(len(data[0]))

            for d in data:
                toadd = zoom(d, scale)
                s.features['downsampled_data']+=list(toadd)



            print len(s.features['downsampled_data'])


            #print s.features

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

if __name__ == '__main__':
    for subject in SUBJECTS:
        preprocess_subject(subject)