#! /usr/bin/python

from __future__ import unicode_literals
from matplotlib import rc
import numpy as np
import cPickle as pickle
from matplotlib import rcParams
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

import seaborn as sns

try:

    phon = pickle.load(open('./phonological-features/phon_pat.pkl', 'r'))
    print np.asarray(phon.values()).shape
except IOError as e:
    print "I/O error({0}): {1}".format(e.errno, e.strerror)
    print 'Try pulling the phonological-features repo, perhaps?'
    exit()

#print R['ivl']# this is the order that the patterns must be sorted into for the dissimilalirty matrix
#st = set(labels)
#print [i for i, e in enumerate( R['ivl']) if e in st]
#a = zip(labels, phon)
#print np.sort(a, order=R['ivl'])
try:
    sem = pickle.load(open('./semantic-features/sem_pat.pkl', 'r'))

    print np.asarray(sem.values()).shape

except IOError as e:
    print "I/O error({0}): {1}".format(e.errno, e.strerror)
    print 'Try pulling the semantic-features repo, perhaps?'
    exit()


for pattern in sem:
  print pattern