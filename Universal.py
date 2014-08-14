__author__ = 'davidnola'

import glob
import cPickle
import scipy
import scipy.io
from math import ceil
from scipy.ndimage.interpolation import zoom
import numpy as np
import itertools
from sklearn.externals import joblib
import sklearn
import sklearn.ensemble
from operator import itemgetter
from random import shuffle
import random
import sys


SEED = 3737

random.seed(SEED)

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
    target_data = np.array([])
    train_data = np.array([])


    segments = np.array([])
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
                temp = s.features['downsampled_data'] + s.features['downsampled_data_1st_deriv'] + s.features['downsampled_data_2nd_deriv'] + s.features['1_hot_identifier'] + s.features['1_hot_channel']
                temp2 = s.features['1_hot_identifier']

                s.features = {}
                s.features['all'] = np.array(temp)
                s.features['1_hot_identifier'] =  np.array(temp2)
                #print s.features['all']
                if 'test' in f:
                    #s.calculate_features()
                    s.data = []
                    np.append(target_data, s)
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
                s.data = np.array([])
                np.append(train_data, s)


    for t in train_data:
        print t.features
        print t.name
        print t.latency


    print "wait ", subject
    #cPickle.dump(train_data, open("/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+subject+'.jpkl', 'wb'),compress = 0, cache_size=1000)
    joblib.dump(train_data, "/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+subject+'.jpkl',compress = 0, cache_size=1000)
    print "finished train"
    #cPickle.dump(target_data, open("/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+subject+'_TEST.jpkl', 'wb'), compress = 0, cache_size=1000)
    joblib.dump(target_data, "/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+subject+'_TEST.jpkl', compress = 0, cache_size=1000)
    print "done ", subject

def fit_random_forests(subject_list):


    (train_clips,target_clips) = load_dataset_pickle(subject_list)

    (train,classes,ids_class,names, uids) = format_dataset(train_clips)

    (target,classes_tgt,ids_class_tgt,names_tgt, uids_tgt) = format_dataset(target_clips)


    fit = train[::2]
    fit_class = classes[::2]

    valid = train[1:][::2]
    valid_class = classes[1:][::2]

    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = 400, n_jobs = 8, verbose = 1, random_state=SEED)
    print "Fitting forests..."
    clf.fit(fit, fit_class)
    results =  clf.predict_proba(valid)
    results_tgt =  clf.predict_proba(target)
    print "SCORE:", clf.score(valid, valid_class)

    (layer_2_features,layer_2_classes,layer_2_ids, layer_2_uids) = generate_second_layer(results, ids_class[1:][::2], names[1:][::2], classes[1:][::2], uids[1:][::2])



    print names_tgt
    (layer_2_features_tgt,layer_2_classes_tgt,layer_2_ids_tgt, layer_2_uids_tgt) = generate_second_layer(results_tgt, ids_class_tgt, names_tgt, classes_tgt, uids_tgt)


    for k in sorted(layer_2_features.keys()):
        v = layer_2_features[k]
        #print [layer_2_classes[k]], layer_2_ids[k]
        layer_2_features[k] = [np.min(v), np.max(v), np.mean(v), np.var(v)] + layer_2_ids[k]

    print "Target data:"
    for k in sorted(layer_2_features_tgt.keys()):
        print k
        v = layer_2_features_tgt[k]
        #print [layer_2_classes_tgt[k]], layer_2_ids_tgt[k]
        layer_2_features_tgt[k] = [np.min(v), np.max(v), np.mean(v), np.var(v)] + layer_2_ids_tgt[k]



    topl = [(layer_2_features[k], layer_2_classes[k]) for k in sorted(layer_2_features.keys())]
    topl_feats = [x[0] for x in topl]
    topl_class = [x[1] for x in topl]
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = 400, n_jobs = 8, verbose = 1, random_state=SEED)
    clf.fit(topl_feats, topl_class)

    topl_tgt = [(layer_2_features_tgt[k], k) for k in sorted(layer_2_features_tgt.keys())]
    topl_feats_tgt = [x[0] for x in topl_tgt]
    topl_class_names = [x[1] for x in topl_tgt]
    result = clf.predict_proba(topl_feats_tgt)

    result = [r[1] for r in result]

    print layer_2_features
    toret = zip(topl_class_names, result)
    print toret
    return toret

#rmemeber : combine our second layers, and also try (stacking second layers - using his as features)
def load_dataset_joblib(subject_list):
    print "loading datasets"
    train_clips = []
    target_clips = []
    for s in subject_list:
        #print s
        train_clips += joblib.load("/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+s+'.jpkl')
        print "Finished loading train"
        target_clips += joblib.load("/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+s+'_TEST.jpkl')
        print "Finished loading target"


    shuffle(train_clips)
    shuffle(target_clips)

    return (train_clips,target_clips)

def load_dataset_pickle(subject_list):
    print "loading datasets"
    sys.stdout.flush()
    train_clips = []
    target_clips = []
    for s in subject_list:
        #print s
        sys.stdout.flush()
        train_clips += cPickle.load(open("/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+s+'.pkl', 'rb'))
        print "Finished loading train"
        sys.stdout.flush()
        target_clips += cPickle.load(open("/Users/davidnola/PyCharm/dnola-summer-research-new/Universal/"+s+'_TEST.pkl', 'rb'))
        print "Finished loading target"
        sys.stdout.flush()


    shuffle(train_clips)
    shuffle(target_clips)

    return (train_clips,target_clips)


def format_dataset(train_clips):
    train = []
    classes = []
    ids_class = []
    uids = []
    names = []
    for t in train_clips:
        train.append(t.features['all'])
        try:
            classes.append(t.seizure)
        except:
            classes.append(-1)

        ids_class.append(t.features['1_hot_identifier'])
        names.append(t.name)
        uids.append(t.id)
    return (train,classes,ids_class,names,uids)

def generate_second_layer(results, ids, names, classes, uids):
    layer_2_features = {}
    layer_2_classes = {}
    layer_2_ids = {}
    layer_2_uids = {}

    v_id_it = iter(ids)
    v_uid_it = iter(uids)
    v_name_it = iter(names)
    v_class_it = iter(classes)

    for r in results:
        next_id = v_id_it.next()
        next_uid = v_uid_it.next()
        next_name = v_name_it.next()
        next_class = v_class_it.next()
        #print next_id, next_name,  r
        if not layer_2_features.has_key(next_name):
            layer_2_features[next_name] = []

        layer_2_features[next_name].append(r[1])
        if layer_2_classes.has_key(next_name):
            assert layer_2_classes[next_name] == next_class

        layer_2_classes[next_name] = next_class
        layer_2_ids[next_name] = next_id
        layer_2_uids[next_name] = next_uid

    return (layer_2_features,layer_2_classes,layer_2_ids, layer_2_uids)

def write_output(data):
    output = "clip,seizure,early\n"

    for d in data:
        output += str(d[0])+","+str(d[1])+","+str(d[1])+"\n"

    #print output
    with file('DistributedSubmitSingle-23-c2.csv', 'w') as f:
        f.write(output)

def generate_layer_1_dict(subject_list):
    print "generating dict"
    sys.stdout.flush()
    (train_clips,target_clips) = load_dataset_pickle(subject_list)

    (train,classes,ids,names, uids) = format_dataset(train_clips)

    (target,classes_tgt,ids_tgt,names_tgt, uids_tgt) = format_dataset(target_clips)

    toret = {}
    feats = iter(train+target)
    ids = iter(ids+ids_tgt)
    uids = iter(uids + uids_tgt)

    print "combining features back together"
    sys.stdout.flush()
    for n in (names+names_tgt):
        f = feats.next()
        uid = uids.next()
        try:
            #print n, toret[n]
            toret[n].append((f,uid))

        except BaseException as e:
            #print e.msg
            toret[n] = [(f,uid)]

    print "sorting combined features"
    sys.stdout.flush()

    for k in sorted(toret.keys()):
        temp = toret[k]
        #print temp
        sorted(temp, key=itemgetter(1))
        toret[k] = []
        for t in temp:
            toret[k]+=t[0]
        print len(toret[k])
    #print toret
    print "done"
    sys.stdout.flush()
    return toret

def generate_layer_2_dict(subject_list):

    sys.stdout.flush()
    (train_clips,target_clips) = load_dataset_pickle(subject_list)

    (train,classes,ids_class,names, uids) = format_dataset(train_clips)

    (target,classes_tgt,ids_class_tgt,names_tgt, uids_tgt) = format_dataset(target_clips)


    fit = train[::2]
    fit_class = classes[::2]

    valid = train[1:][::2]
    valid_class = classes[1:][::2]


    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = 500, n_jobs = 8, verbose = 1, random_state=SEED)
    print "Fitting forests..."
    clf.fit(fit, fit_class)
    results =  clf.predict_proba(valid)
    results_tgt =  clf.predict_proba(target)
    print "SCORE:", clf.score(valid, valid_class)

    print "generating first half"
    (layer_2_features,layer_2_classes,layer_2_ids, layer_2_uids) = generate_second_layer(results, ids_class[1:][::2], names[1:][::2], classes[1:][::2], uids[1:][::2])

    first = layer_2_features
    first_ids = layer_2_ids

    fit = train[1::2]
    fit_class = classes[1::2]

    valid = train[:][::2]
    valid_class = classes[:][::2]

    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = 500, n_jobs = 8, verbose = 1, random_state=SEED)
    print "Fitting forests..."
    clf.fit(fit, fit_class)
    results =  clf.predict_proba(valid)
    results_tgt =  clf.predict_proba(target)
    print "SCORE:", clf.score(valid, valid_class)

    train = []
    fit = []
    fit_class = []
    valid = []
    valid_class = []

    # print "generating second half"
    # (layer_2_features,layer_2_classes,layer_2_ids, layer_2_uids) = generate_second_layer(results, ids_class[:][::2], names[:][::2], classes[:][::2], uids[:][::2])
    #
    # second = layer_2_features
    # second_ids = layer_2_ids
    #
    #
    #
    # #print names_tgt

    print "generating target layer"
    (layer_2_features_tgt,layer_2_classes_tgt,layer_2_ids_tgt, layer_2_uids_tgt) = generate_second_layer(results_tgt, ids_class_tgt, names_tgt, classes_tgt, uids_tgt)

    # print "lengths: ", len(first.keys()), len(first.keys())
    #
    # for k in second.keys():
    #     try:
    #         first[k]+=second[k]
    #     except:
    #         first[k] = second[k]
    #
    # first_ids.update(second_ids)

    print "transforming layer"
    for k in sorted(first.keys()):
        v = first[k]
        #print [layer_2_classes[k]], layer_2_ids[k]

        #print "v: "
        #print list(v)
        #print np.min(v)


        #print [np.min(v), np.max(v), np.mean(v), np.var(v)]
        first[k] = [np.min(v), np.max(v), np.mean(v), np.var(v)] + first_ids[k]

    print "Target data transform:"
    for k in sorted(layer_2_features_tgt.keys()):
        #print k
        v = layer_2_features_tgt[k]
        #print [layer_2_classes_tgt[k]], layer_2_ids_tgt[k]
        layer_2_features_tgt[k] = [np.min(v), np.max(v), np.mean(v), np.var(v)] + layer_2_ids_tgt[k]



    for t in [n.name for n in train]:
        if not t in first:
            print "NO:", t


    first.update(layer_2_features_tgt)

    return first # PROBLEM: We need a dict of EVERYTHING on this layer - no wait - should skip ones we dont have

if __name__ == '__main__':

    preprocess_subject('Dog_1')

    #     pass
    #data = fit_random_forests(SUBJECTS[:])
    #write_output(data)

    #print generate_layer_1_dict(SUBJECTS[0:1])
    #generate_layer_2_dict(SUBJECTS[2:3])

    print "DONE"

