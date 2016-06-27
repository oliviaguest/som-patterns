#! /usr/bin/python

from __future__ import unicode_literals
from matplotlib import rc
import numpy as np
import copy as cp
import cPickle as pickle
from matplotlib import rcParams
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

import seaborn as sns


def create_heatmap(X, labels, filename):
    # generate the linkage matrix
    corrmat = 1-pairwise_distances(X, metric="correlation")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 9))

    # Draw the heatmap using seaborn
    sns.heatmap(corrmat, vmax=1, vmin=-1, square=True, xticklabels = labels, yticklabels = labels)

    # Use matplotlib directly to emphasize known networks
    #networks = corrmat.columns.get_level_values("network")
    #for i, network in enumerate(networks):
      #if i and network != networks[i - 1]:
          #ax.axhline(len(networks) - i, c="w")
          #ax.axvline(i, c="w")
    f.tight_layout()

    # This sets the yticks "upright" with 0, as opposed to sideways with 90.
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.savefig('./fig/'+filename+'_heatmap.png', bbox_inches='tight')
    # plt.savefig('./fig/'+filename+'_heatmap.pdf', bbox_inches='tight')
    del f, ax




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


#for the (dis)similalrity matrix aka heatmap we need the semantic reps in order
#otherwise the inherent categorical structure won't show
pkl_file = open('./semantic-features/resources/Tyler.pkl', 'rb')
_, _, concepts, _, _ = pickle.load(pkl_file)
X = []
labels = []
for i in concepts:
  #print i
  #print
  if i in phon:
    X.append(phon[i])
    labels.append(i)

sem_cpy = cp.deepcopy(sem)
distortions = range(1,6)
sem_ordered = []
sem_ordered_labels = []
for label in labels:
    for distortion in distortions:
        sem_cpy[label+str(distortion)] = sem[label] + np.random.normal(0, 0.1*distortion, len(sem[label]))
        sem_ordered.append(sem_cpy[label+str(distortion)])
        sem_ordered_labels.append(label+str(distortion))
    # print label+'1 ', sem_cpy[label+'1']
phon_cpy = cp.deepcopy(phon)
distortions = range(1,6)
phon_ordered = []
phon_ordered_labels = []
for label in labels:
    for distortion in distortions:
        phon_cpy[label+str(distortion)] = phon[label] + np.random.normal(0, 0.1*distortion, len(phon[label]))
        phon_ordered.append(phon_cpy[label+str(distortion)])
        phon_ordered_labels.append(label+str(distortion))
    print label+'1 ', phon_cpy[label+'1']

create_heatmap(sem_ordered, sem_ordered_labels, 'sem_distortions')
create_heatmap(phon_ordered, phon_ordered_labels, 'phon_distortions')

pickle.dump([sem_ordered, phon_ordered, sem_ordered_labels], open('./distortions.pkl', 'w'))
