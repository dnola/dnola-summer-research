__author__ = 'davidnola'

import cPickle
from Basic import EEGSegment
import glob
import scipy.io
from multiprocessing import *
import numpy as np
SUBJECTS = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']

#SUBJECTS = ['Dog_1']



def add_total_channel_variance(subject, location):
    clips = [];
    test_data = [];
    location = location+subject+'/*.mat'
    print location
    f = iter(glob.glob(location))
    for fpkl in glob.glob('*.pkl'):
        pkl = cPickle.load(open(fpkl, 'rb'))
        for s in pkl:
            print s.name
            print f.next()
            print




def pickle_dataset(subject, location):
    clips = [];
    test_data = [];
    location = location+subject+'/*.mat'
    print location
    for f in glob.glob(location):
        print f
        #sub_file = open(f , 'r')
        #print f

        s = EEGSegment()
        matfile = scipy.io.loadmat(f)
        s.data = matfile['data']
        s.frequency = matfile['freq']
        s.name = f[f.rindex('/')+1:]

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
    print "cleared"
    cPickle.dump(clips, open(subject+'.pkl', 'wb'))
    cPickle.dump(test_data, open(subject+'_TEST.pkl', 'wb'))

def train_op(clip, mat, f):
    add_time_variance_feature(clip, mat)

    return


def add_time_variance_feature(clip, mat):
    data = mat['data']
    for i in range(10):
        clip.features.pop('variance_at_time_'+str(i), None)

    toadd = []
    toadd_delta = []

    for d in data:
        size = len(d)/10 + 1
        for i in range(10):
            chunk = d[i*size : (i+1)*size]
            toadd.append(np.var(chunk))
            if i!=0:
                toadd_delta.append(toadd[-1] - toadd[-2])

    clip.features['variance_over_time_10'] = toadd
    clip.features['delta_var_over_10'] = toadd_delta

    print clip.features


def early_seizure_naming(clip, mat ,f):
    lty = -1

    if mat.has_key('latency'):
        lty = mat['latency']


    print clip.latency, lty
    #print f

    clip.name = f
    if clip.latency <= 15 and clip.seizure == 1:
        clip.seizure_early = 1
    else:
        clip.seizure_early = 0


def test_op(clip, mat):
    add_time_variance_feature(clip, mat)
    return



def kickoff_subject(subject):
    files = glob.glob('/Users/davidnola/Machine_Learning_Research/Data/'+ subject+'/*.mat')
    f_iter = iter(files)



    test = cPickle.load(open(subject+'_TEST.pkl', 'rb'))
    train = cPickle.load(open(subject+'.pkl', 'rb'))


    print len(test)

    for t in train:
        f = f_iter.next()

        matfile = scipy.io.loadmat(f)
        train_op(t, matfile, f)

    for t in test:
        f = f_iter.next()
        matfile = scipy.io.loadmat(f)
        test_op(t, matfile)



    print "wait ", subject
    cPickle.dump(train, open(subject+'.pkl', 'wb'))
    cPickle.dump(test, open(subject+'_TEST.pkl', 'wb'))
    print "done ", subject



if __name__ == '__main__':
    p = []
    for s in SUBJECTS:
        p.append(Process(target=kickoff_subject, args=(s,)))

    for proc in p:
        proc.start()

    for proc in p:
        proc.join()





