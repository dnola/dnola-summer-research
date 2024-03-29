__author__ = 'davidnola'

import cPickle
from Basic import EEGSegment
import glob
import scipy.io
import scipy.misc
from multiprocessing import *
import numpy as np
import itertools
import scipy
import scipy.ndimage
import scipy.ndimage.interpolation
import Universal
from Universal import UniversalSegment
import sys
from itertools import chain, combinations
SUBJECTS = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2','Patient_3','Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']

#SUBJECTS = ['Dog_1']


"""
def add_total_channel_variance(subject, location):
    clips = [];
    test_data = [];
    location = location+subject+'/*.mat'
    print location
    f = iter(glob.glob(location))
    for fpkl in glob.glob(subject+'*.pkl'):
        print "loaded: ", fpkl
        pkl = cPickle.load(open(fpkl, 'rb'))
        for s in pkl:
            print s.name
            print f.next()
            print
"""

def combine_pickles():
    second_iter = itertools.cycle(glob.glob('Second/*.pkl'))

    for first in glob.glob('First/*.pkl'):
        final_pkl = []
        second = second_iter.next()

        f_pkl = cPickle.load(open(first, 'rb'))
        s_pkl = cPickle.load(open(second, 'rb'))
        sec = itertools.cycle(s_pkl)
        for f in f_pkl:
            s = sec.next()
            while f.name!=s.name:
                print "Bad", f.name, s.name
                s = sec.next()


            seg = EEGSegment()
            seg.frequency = f.frequency
            seg.name = f.name
            seg.latency = f.latency
            seg.seizure = f.seizure
            seg.features = {}
            try:
                seg.features = f.features

                for k in s.features.keys():
                    print k
                    seg.features[k] = s.features[k]
            except:
                print "Failed to set features"

            final_pkl.append(seg)
            #print seg.features
        cPickle.dump(final_pkl, open(first[first.rfind('/')+1:], 'wb'),-1)


def combine_features(subject):
    for fpkl in glob.glob(subject+'*.pkl'):
        clips = [];
        print "loading: ", fpkl
        pkl = cPickle.load(open(fpkl, 'rb'))
        sys.stdout.flush()

        for s in pkl:
            toadd = []
            for k in s.features.keys():
                toadd+=s.features[k]

            s.features['all_features'] = toadd
            clips.append(s)
            #print s.features
        cPickle.dump(clips, open(fpkl, 'wb'),-1)

def add_feature(subject, location, feature, fidelity = 0):
    location = location+subject+'/*.mat'
    print location
    f = iter(glob.glob(location))
    for fpkl in glob.glob(subject+'*.pkl'):
        clips = [];
        print "loaded: ", fpkl
        pkl = cPickle.load(open(fpkl, 'rb'))
        for s in pkl:
            if '1.' in s.name:
                print s.name, fpkl
            mat = scipy.io.loadmat(f.next())
            s.data = mat['data']
            if fidelity == 0:
                feature(s, mat)
            else:
                feature(s, mat, fidelity)
            s.data=[]

            #print s.features
            clips.append(s)
        cPickle.dump(clips, open(fpkl, 'wb'),-1)

def powerset_features(subject):
    for fpkl in glob.glob(subject+'*.pkl'):
        clips = [];
        print "loading: ", fpkl
        pkl = cPickle.load(open(fpkl, 'rb'))
        sys.stdout.flush()

        for s in pkl:
            toadd = []
            for combo in powerset_generator(s.features.keys()):
                combo = list(combo)
                if len(combo) < 1:
                    continue
                s.features[str(combo)]=[]
                #print "combo", combo
                for c in combo:
                    #print c
                    s.features[str(combo)]+=s.features[c]


            clips.append(s)
            #print "features", s.features
        cPickle.dump(clips, open(fpkl, 'wb'),-1)




def powerset_generator(i):
    for subset in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
        yield set(subset)



def add_feature_universal(subject, data):
    for fpkl in glob.glob(subject+'*.pkl'):
        clips = [];
        print "loading: ", fpkl
        pkl = cPickle.load(open(fpkl, 'rb'))
        sys.stdout.flush()

        for s in pkl:
            s.features['universal_lower'] = []
            #print s.features
            #print data[s.name]
            s.features['universal_lower'] = data[s.name]
            clips.append(s)
            #print s.features
        cPickle.dump(clips, open(fpkl, 'wb'),-1)


#####################################################################################################################

def channel_downsample(seg, mat, fidelity=100):
    feat = 'channel_downsample_x'+str(fidelity)
    seg.features[feat] = []

    chan_count =  len (mat['data'])
    scale = fidelity/float(len(mat['data'][0]))
    print scale

    for d in mat['data']:
        toadd = scipy.ndimage.interpolation.zoom(d, scale)
        seg.features[feat]+=list(toadd)

    print seg.name, "done "



# def channel_downsample_multi(seg, mat, fidelity=100):
#     feat = 'channel_downsample_x'+str(fidelity)
#     seg.features[feat] = []
#
#     chan_count =  len (mat['data'])
#     scale = fidelity/float(len(mat['data'][0]))
#     print scale
#
#     for d in mat['data']:
#         toadd = scipy.ndimage.interpolation.zoom(d, scale)
#         seg.features[feat]+=list(toadd)
#
#     print seg.name, "done "


    #resized = scipy.misc.imresize(mat['data'], ( chan_count, fidelity) )



    # for r in resized:
    #     seg.features[feat]+=list(r)

    #print seg.features


def channel_variance_ratios(seg, mat, fidelity=5):
    feat = 'channel_variance_ratios_x'+str(fidelity)
    seg.features[feat] = []
    for d in mat['data']:
        ch_var = np.var(d)
        seg_size = len(d)/int(fidelity)
        #print d
        for i in range(fidelity):
            toadd = np.var(d[i*seg_size:(i+1)*seg_size]) / ch_var
            #print "add:", toadd
            seg.features[feat].append(toadd)
    #print seg.features

def channel_FFT_components(seg, mat, fidelity=9):
    feat = 'FFT_values_x'+str(fidelity)
    seg.features[feat] = []
    for d in mat['data']:
        fft = np.fft.fft(d, fidelity)
        fft_freq = []
        #print fft
        for f in fft:
            fft_freq.append((f.imag ** 2 + f.real ** 2) ** .5)

        seg.features[feat]+=fft_freq
    #print seg.features



def total_variance_ratios(seg, mat, fidelity=5):
    feat = 'total_variance_ratios_x'+str(fidelity)
    ch_var = []
    for d in mat['data']:
        ch_var.append(np.var(d))
    ch_var = np.mean(ch_var)

    seg.features[feat] = []
    for d in mat['data']:
        seg_size = len(d)/int(fidelity)
        #print d
        for i in range(fidelity):
            toadd = np.var(d[i*seg_size:(i+1)*seg_size]) / ch_var
            #print "add:", toadd
            seg.features[feat].append(toadd)
    #print seg.features


def channel_sigmas(pkl, mat, fidelity=5):
    cursig = 0
    for siglevel in ['channel_1sig_times_exceeded_delta', 'channel_2sig_times_exceeded_delta']:
        pkl.features[siglevel] = []
        cursig+=1
        iter = 0.0
        for d in mat['data']:
            stddev = np.std(d)
            mean = np.mean(d)
            prior = d[0]
            exceeded_first = 0
            for x in d[1:len(d)/2]:
                if prior < (cursig*stddev + mean) and x > (cursig*stddev + mean):
                    exceeded_first+=1
                prior = x

            pkl.features[siglevel].append(exceeded_first)



            prior = d[len(d)/2]
            exceeded = 0
            for x in d[len(d)/2+1:]:
                if prior < (cursig*stddev + mean) and x > (cursig*stddev + mean):
                    exceeded+=1
                prior = x

            pkl.features[siglevel].append(exceeded)
            pkl.features[siglevel].append(exceeded - exceeded_first)

    pkl.data = []
    print pkl.features


def add_total_channel_variance(subject, location):

    location = location+subject+'/*.mat'
    print location
    f = iter(glob.glob(location))
    for fpkl in glob.glob(subject+'*.pkl'):
        clips = [];
        print "loaded: ", fpkl
        pkl = cPickle.load(open(fpkl, 'rb'))
        for s in pkl:
            print s.name
            mat = scipy.io.loadmat(f.next())
            s.data = mat['data']
            s.features['channel_variances'] = []
            for d in s.data:
                x = np.var(d)
                s.features['channel_variances'].append(x)

            print s.features
            s.data=[]
            clips.append(s)
        cPickle.dump(clips, open(fpkl, 'wb'),-1)

def add_channel_variance_change(subject, location):

    location = location+subject+'/*.mat'
    print location
    f = iter(glob.glob(location))
    for fpkl in glob.glob(subject+'*.pkl'):
        clips = [];
        print "loaded: ", fpkl
        pkl = cPickle.load(open(fpkl, 'rb'))
        for s in pkl:
            print s.name
            mat = scipy.io.loadmat(f.next())
            s.data = mat['data']

            toadd = []
            toadd_delta = []

            s.features['channel_variances_quarter'] = []
            s.features['channel_variances_delta'] = []
            for d in s.data:
                size = len(d)/4 + 1
                for i in range(4):
                    chunk = d[i*size : (i+1)*size]
                    toadd.append(np.var(chunk))
                    if i!=0:
                        toadd_delta.append(toadd[-1] - toadd[-2])

            s.features['channel_variances_quarter'] = toadd
            s.features['channel_variances_delta'] = toadd_delta

            print s.features
            s.data=[]
            clips.append(s)
        cPickle.dump(clips, open(fpkl, 'wb'),-1)

#new sig : just 1 and 2 levels, do first half, second half, and deltas
#new varsplit: 4 segments, 3 deltas


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
    print "cleared"
    cPickle.dump(clips, open(subject+'.pkl', 'wb'),-1)
    for t in test_data:
        if 'ictal' in t.name:
            print "BAD CLIP IN TEST DATA"
            exit(1)
    for t in clips:
        if 'test' in t.name:
            print "BAD CLIP IN TRAIN DATA"
            exit(1)

    cPickle.dump(test_data, open(subject+'_TEST.pkl', 'wb'),-1)

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
    cPickle.dump(train, open(subject+'.pkl', 'wb'),-1)
    cPickle.dump(test, open(subject+'_TEST.pkl', 'wb'),-1)
    print "done ", subject



if __name__ == '__main__':
    p = []
    # for s in SUBJECTS:
    #     p.append(Process(target=kickoff_subject, args=(s,)))
    #
    # for proc in p:
    #     proc.start()
    #
    # for proc in p:
    #     proc.join()

