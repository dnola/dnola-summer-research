__author__ = 'davidnola'

import glob
import cPickle
import scipy
import scipy.io
from math import ceil
from scipy.ndimage.interpolation import zoom
import numpy as np
import itertools

import sklearn
import sklearn.ensemble

from random import shuffle

class UniversalSegment:
    def __init__(self):
        pass


SUBJECTS = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']


one_hots = {}
it = iter(range(12))
for s in SUBJECTS:
    next = it.next()
    one_hots[s] = [1  if x == next else 0 for x in range(12)]

#print one_hots

one_hots_channel = {}
it = iter(range(100))
for s in range(100):
    next = it.next()
    one_hots_channel[s] = [1  if x == next else 0 for x in range(100)]

#print one_hots_channel


def preprocess_subject(subject):
    files = glob.glob('/Users/davidnola/Data/'+ subject+'/*.mat')
    f_iter = iter(files)



    #test = cPickle.load(open(subject+'_TEST.pkl', 'rb'))
    #train = cPickle.load(open(subject+'.pkl', 'rb'))
    target_data = []
    train_data = []


    segments = []
    id = itertools.count()

    for f in files:
        print f
        matfile = scipy.io.loadmat(f)

        data_file = matfile['data']
        #print "Channel count: ", len(data_file)
        channel_num = itertools.count()
        for channel in data_file:
            cur_num = channel_num.next()
            for i in range(4):

                s = UniversalSegment()
                size = ceil(len(channel)/4.0)

                data = channel[ i*size : (i+1)*size]


                s.data = data
                s.id = id.next()
                s.frequency = matfile['freq']
                s.name = f[f.rindex('/')+1:]
                s.features={}



                scale = 400/float(len(channel))
                s.features['downsampled_data'] = zoom(data, scale)



                s.features['downsampled_data_1st_deriv'] = np.diff(s.features['downsampled_data'])
                s.features['downsampled_data_2nd_deriv'] = np.diff(s.features['downsampled_data_1st_deriv'])
                s.features['1_hot_identifier'] = one_hots[subject]
                s.features['1_hot_channel'] = one_hots_channel[cur_num]
                #print s.features



                s.features['downsampled_data'] = [np.percentile(s.features['downsampled_data'], x * 10) for x in range(11)]
                s.features['downsampled_data_1st_deriv'] = [np.percentile(s.features['downsampled_data_1st_deriv'], x * 10) for x in range(11)]
                s.features['downsampled_data_2nd_deriv'] = [np.percentile(s.features['downsampled_data_2nd_deriv'], x * 10) for x in range(11)]
                #print s.features
                s.features['all'] = s.features['downsampled_data'] + s.features['downsampled_data_1st_deriv'] + s.features['downsampled_data_2nd_deriv'] + s.features['1_hot_identifier'] + s.features['1_hot_channel']
                #print s.features['all']
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
    cPickle.dump(train_data, open("Universal/"+subject+'.pkl', 'wb'))
    print "finished train"
    cPickle.dump(target_data, open("Universal/"+subject+'_TEST.pkl', 'wb'))
    print "done ", subject

def fit_random_forests(subject_list):
    train_clips = []
    target_clips = []
    for s in subject_list:
        print s
        train_clips += cPickle.load(open("Universal/"+s+'.pkl', 'rb'))
        print "Finished loading train"
        target_clips += cPickle.load(open("Universal/"+s+'_TEST.pkl', 'rb'))
        print "Finished loading target"


    shuffle(train_clips)
    shuffle(target_clips)

    train = []
    classes = []
    ids = []
    for t in train_clips:
        train.append(t.features['all'])
        classes.append(t.seizure)
        ids.append(t.name)

    #print train

    fit = train[::2]
    fit_class = classes[::2]

    valid = train[1:][::2]
    valid_class = classes[1:][::2]

    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = 30)
    print "Fitting forests..."
    clf.fit(fit, fit_class)
    results =  clf.predict_proba(valid)
    print "SCORE:", clf.score(valid, valid_class)

    layer_2_features = {}

    v_it = iter(ids[1:][::2])
    for r in results:
        next = v_it.next()
        print next, r

        #if not layer_2_features.has_key(next):
        #    layer_2_features[next] =






if __name__ == '__main__':
    for subject in SUBJECTS[:]:
        #preprocess_subject(subject)
        pass
    fit_random_forests(SUBJECTS[0:1])