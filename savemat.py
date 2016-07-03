#! /usr/bin/python

import numpy as np
import cPickle as pkl
import scipy.io as sio

try:

    sem = pkl.load(open('./semantic-features/sem_pat.pkl', 'r'))
    print np.asarray(sem.values()).shape
except IOError as e:
    print "I/O error({0}): {1}".format(e.errno, e.strerror)
    print 'Try pulling the semantic-features repo, perhaps?'
    exit()
try:

    phon = pkl.load(open('./phonological-features/phon_pat.pkl', 'r'))
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
  pkl_file = open('./semantic-features/resources/Tyler.pkl', 'rb')
  patterns, _, labels, _, _ = pkl.load(pkl_file)

except IOError as e:
    print "I/O error({0}): {1}".format(e.errno, e.strerror)
    print 'Try pulling the semantic-features repo, perhaps?'
    exit()

#for the (dis)similalrity matrix aka heatmap we need the semantic reps in order
#otherwise the inherent categorical structure won't show
pkl_file = open('./semantic-features/resources/Tyler.pkl', 'rb')
_, _, concepts, _, _ = pkl.load(pkl_file)
print type(concepts)
Phon = []
Sem = []
labels = []
for i in concepts:
#print i
#print
    if i == 'telephone':
        i = 'phone'
    if i in sem:
      Phon.append(phon[i])
      Sem.append(sem[i])
      labels.append(i)
print len(Sem), len(Phon)
sio.savemat('som_patterns.mat', {'phon':Phon, 'sem': Sem, 'labels': labels})
