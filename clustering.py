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




#first we do the phonology
if True:
      X = np.asarray(phon.values())
      labels = phon.keys()
      
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
            
      # generate the linkage matrix
      Z = sch.linkage(X, 'ward')
      c, coph_dists = sch.cophenet(Z, pdist(X, 'jaccard'))

      c, coph_dists = sch.cophenet(Z, pdist(X))
      # cophenetic correlation coefficient of clustering
      print 'Cophenetic correlation coefficient of clustering (the closer to 1, the better):', c

      # calculate full dendrogram
      fig = plt.figure(figsize=(20, 10))
      ax = fig.add_subplot(111)
      plt.title('Hierarchical Clustering Dendrogram')
      plt.xlabel('Phonological Representations of Words')
      plt.ylabel('Jaccard Distance')
      R = sch.dendrogram(
          Z,
          leaf_rotation=90,  # rotates the x axis labels
          leaf_font_size=16.,  # font size for the x axis labels
          labels = labels,
          )
      ax.tick_params(labelsize=10)

      fig.savefig('./fig/word_dendrogram.png', bbox_inches='tight')
      fig.savefig('./fig/word_dendrogram.pdf', bbox_inches='tight')

      #PCA time
      pca = PCA(n_components=2)
      X_r = pca.fit(X).transform(X)
      PCA(copy=True, n_components=2, whiten=False)
      print(pca.explained_variance_ratio_) 

      fig = plt.figure(figsize=(20, 18))
      ax = fig.add_subplot(111)
      ax.scatter(X_r[:, 0], X_r[:, 1], c='w', label='Word', marker='o', s = 75, alpha = 0.8 )
      ax.set_ylim([-3.3, 2])
      for i, txt in enumerate(labels):
          ax.annotate(txt, (X_r[i, 0], X_r[i, 1]), horizontalalignment='center', verticalalignment='center',size = 14)
      plt.legend()
      plt.title('PCA for the Phonological Representations of Words')
      plt.xlabel('First Principal Component (explained variance ratio = '+str(np.around(pca.explained_variance_ratio_[0], decimals = 3))+')')
      plt.ylabel('Second Principal Component (explained variance ratio = '+str(np.around(pca.explained_variance_ratio_[1], decimals = 3))+')')

      fig.savefig('./fig/word_pca.png', bbox_inches='tight')
      fig.savefig('./fig/word_pca.pdf', bbox_inches='tight')

      corrmat = 1-pairwise_distances(X, metric="cosine")

      # Set up the matplotlib figure
      f, ax = plt.subplots(figsize=(12, 9))

      # Draw the heatmap using seaborn
      sns.heatmap(corrmat, vmax=.8, square=True, xticklabels = labels, yticklabels = labels)

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
      plt.savefig('./fig/word_heatmap.png', bbox_inches='tight')
      plt.savefig('./fig/word_heatmap.pdf', bbox_inches='tight')

#now we do the perceptual/semantic

X = np.asarray(sem.values())
labels = sem.keys()

      
#for the (dis)similalrity matrix aka heatmap we need the semantic reps in order
#otherwise the inherent categorical structure won't show
pkl_file = open('./semantic-features/resources/Tyler.pkl', 'rb')
_, _, concepts, _, _ = pickle.load(pkl_file)
X = []
labels = []
#counter = 0
for i in concepts:
  if i == 'telephone': #Themis renamed this
    i = 'phone'
  if i in sem:
    X.append(sem[i])
    labels.append(i)   
    #counter += 1


pca = PCA(n_components=396)
X_r = pca.fit(X).transform(X)
X =X_r      
# generate the linkage matrix
Z = sch.linkage(X, 'ward')
c, coph_dists = sch.cophenet(Z, pdist(X, 'jaccard'))

c, coph_dists = sch.cophenet(Z, pdist(X))
# cophenetic correlation coefficient of clustering
print 'Cophenetic correlation coefficient of clustering (the closer to 1, the better):', c

# calculate full dendrogram
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Perceptual Representations of Concepts')
plt.ylabel('Jaccard Distance')
R = sch.dendrogram(
    Z,
    leaf_rotation=90,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    labels = labels,
    )
ax.tick_params(labelsize=10)

fig.savefig('./fig/percept_dendrogram.png', bbox_inches='tight')
fig.savefig('./fig/percept_dendrogram.pdf', bbox_inches='tight')




#PCA time
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_) 

fig = plt.figure(figsize=(20, 18))
ax = fig.add_subplot(111)
ax.scatter(X_r[:, 0], X_r[:, 1], c='w', label='Word', marker='o', s = 75, alpha = 0.8 )
#ax.set_ylim([-2, 3])
for i, txt in enumerate(labels):
    ax.annotate(txt, (X_r[i, 0], X_r[i, 1]), horizontalalignment='center', verticalalignment='center',size = 14)
plt.legend()
plt.title('PCA for the Representations of Percepts')
plt.xlabel('First Principal Component (explained variance ratio = '+str(np.around(pca.explained_variance_ratio_[0], decimals = 3))+')')
plt.ylabel('Second Principal Component (explained variance ratio = '+str(np.around(pca.explained_variance_ratio_[1], decimals = 3))+')')

fig.savefig('./fig/percept_pca.png', bbox_inches='tight')
fig.savefig('./fig/percept_pca.pdf', bbox_inches='tight')



corrmat = 1-pairwise_distances(X, metric="cosine")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True, xticklabels = labels, yticklabels = labels)

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
plt.savefig('./fig/percept_heatmap.png', bbox_inches='tight')
plt.savefig('./fig/percept_heatmap.pdf', bbox_inches='tight')