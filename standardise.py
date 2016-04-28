import numpy as np
import cPickle as pkl





#print R['ivl']# this is the order that the patterns must be sorted into for the dissimilalirty matrix
#st = set(labels)
#print [i for i, e in enumerate( R['ivl']) if e in st]
#a = zip(labels, phon)
#print np.sort(a, order=R['ivl'])
try:
  pkl_file = open('./semantic-features/resources/Tyler.pkl', 'rb')
  patterns, _, labels, _, _ = pickle.load(pkl_file)

except IOError as e:
    print "I/O error({0}): {1}".format(e.errno, e.strerror)
    print 'Try pulling the semantic-features repo, perhaps?'
    exit()
labels[list(labels).index("telephone")] = 'phone'
print labels

#create semantic patterns with the same labels
#recall intersection means some get left out
#get the indices of words that are in both word lists
st = set(phon_dict.keys())
#Tyler calls telephone 'telephone' Themis calls in 'phone'

keep_word = [i for i, e in enumerate(labels) if e in st]
delete_word = [i for i, e in enumerate(labels) if e not in st]
print keep_word, delete_word
#delete words I do not need
#recall we calculated this by finding the words
#that are not in the Tyler CDI subset, using their index
patterns = np.delete(patterns, delete_word, 0)

print len(labels)
#and we can also grab the correct labels per word
labels = np.asarray(labels)[keep_word]
print len(labels)

#and create a dictionary for ease of access
phon_dict = dict(zip(labels, phon))


sem_dict = dict(zip(labels, patterns))
pickle.dump(sem_dict, open('sem_pat.pkl', 'w'))





phon = pkl.load(open('phon_pat.pkl', 'r'))
print np.asarray(phon.values()).shape

on = []
for v in phon.values():
  on.append(sum(v))
print np.mean(on)

sem = pkl.load(open('sem_pat.pkl', 'r'))
print np.asarray(sem.values()).shape

on = []
for v in sem.values():
  on.append(sum(v))
print np.mean(on)

assert phon.keys() == sem.keys()


sem_val = np.asarray(sem.values())

#this will keep track of which columns will be kept
keep_col = np.zeros(sem_val.shape[1])
keep_col.fill(False) #assume none will be kept
#for every column
for s in range(0, sem_val.shape[1]):
  #for every pattern
  for p in range(0, sem_val.shape[0]):
     if not (sem_val[p][s] == 0).all():
        #print sem[p][s]
        keep_col[s] = True
  #print keep_col[s]
        
#print keep_col

#we just checked every pattern
#so we can now nuke clumns if required
sem_val = np.delete(sem_val, np.where(keep_col == False), 1)
print sem_val.shape
