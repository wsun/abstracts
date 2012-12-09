# CS205 Final Project
# Janet Song and Will Sun
#
# K-means clustering, using NLTK
# 
# Evaluate different representations and similarity measures

from nltk.cluster import KMeansClusterer
from collections import defaultdict
import numpy as np
import math

def euclidean(u, v):
    ''' Return euclidean distance between vectors u and v. '''
    diff = u - v
    return math.sqrt(np.dot(diff, diff))

def cosine(u, v):
    ''' Return 1 minus the cosine between vectors u and v. '''
    return 1.0 - (np.dot(u,v) / 
                    (math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v,v))))

def jaccard(u, v):
    ''' Return 1 minus the Jaccard similarity between vectors u and v. '''
    setU = 0
    setV = 0
    intersection = 0
    for i in xrange(len(u)):
        if u[i] != 0:
            setU += 1   # in u
        if v[i] != 0:
            setV += 1   # in v
        if u[i] != 0 and v[i] != 0:
            intersection += 1  # in both u and v
    return 1.0 - (float(intersection) / (setU + setV - intersection))

def sumdistance(vectors, clusters, means):
    ''' 
    Calculate total euclidean distances between every vector and its
    assigned mean.
    '''
    total = 0
    for i, v in enumerate(vectors):
        cluster = clusters[i]
        mean = means[cluster]
        total += euclidean(v, mean)
    return total

def purity(clusters, labels, k):
    ''' 
    Evaluate the coherence of a cluster; assume that a cluster is labeled with
    the dominant category contained within and measure how accurate that is.
    '''
    # track distribution of labels in each cluster
    cluster_labels = [ defaultdict(int) for i in xrange(k) ]
    # total length of docs considered
    num = len(clusters)
    
    # loop through all vectors
    for i in xrange(num):
        cluster = clusters[i]
        cluster_labels[cluster][labels[i]] += 1     # increment label count

    # track count of documents in cluster belonging to dominant label
    cluster_dominant = []
    for i in xrange(k):
        dominant = 0
        for key, v in cluster_labels[i].iteritems():
            if v > dominant:
                dominant = v
        cluster_dominant.append(dominant)

    # compute weighted average of all purities
    purity = 0
    for i in xrange(k):
        purity +=  cluster_dominant[i]
    purity *= (1.0 / num)
    return purity


def entropy(clusters, labels, k):
    ''' Evaluate the distribution of categories within a cluster. '''
    # track distribution of labels in each cluster
    cluster_labels = [ defaultdict(int) for i in xrange(k) ]
    # track total number of items per cluster
    cluster_count = np.zeros(k, dtype=np.float64)
    # track all docs
    total = len(clusters)
    
    # loop through all vectors
    for i in xrange(total):
        cluster = clusters[i]
        cluster_count[cluster] += 1.0               # increment cluster count
        cluster_labels[cluster][labels[i]] += 1     # increment label count

    # compute weighted average of all entropies
    entropy = 0
    for i in xrange(k):
        cluster_entropy = 0
        for key, v in cluster_labels[i].iteritems():
            cluster_entropy += (v / cluster_count[i]) * math.log(v / cluster_count[i])
        entropy += cluster_entropy * (-1.0 / math.log(k)) * (cluster_count[i] / total)
    return entropy

def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def randindex(clusters, labels, k):
    ''' 
    Evaluate the percentage of correct clustering with the Rand index. 
    RI = (TP + TN) / (TP + FP + FN + TN)
    '''

    # track distribution of labels in each cluster
    cluster_labels = [ defaultdict(int) for i in xrange(k) ]
    # track total number of items per cluster
    cluster_count = np.zeros(k)
    # track distribution of documents throughout clusters, given a label
    label_loc = {}
    for l in labels:    # initialize
        label_loc.setdefault(l, np.zeros(k))
    
    # loop through all vectors
    for i in xrange(len(clusters)):
        cluster = clusters[i]
        cluster_count[cluster] += 1                 # increment cluster count
        cluster_labels[cluster][labels[i]] += 1     # increment label tracker
        label_loc[labels[i]][cluster] += 1          # increment cluster tracker

    # loop through clusters, compute all positives and true positives
    ##### all positives: any pair of documents in same cluster
    ##### all negatives: any pair of documents in different clusters
    ##### true positives: any pair of documents in same cluster with same label
    ##### false negatives: any pair of documents in diff cluster with same label
    pos = 0
    neg = 0
    tp = 0
    fn = 0
    label_clusters = label_loc.items()
    for i in xrange(k):
        pos += nCr(cluster_count[i], 2)
        neg += cluster_count[i] * sum(cluster_count[i+1:])
        for key, v in cluster_labels[i].iteritems():
            if v > 1: 
                tp += nCr(v, 2)

        samepairs = 0
        labelcount = 0
        for v in label_clusters[i][1]:
            labelcount += v
            if v > 1:
                samepairs += nCr(v, 2)
        fn += nCr(labelcount, 2) - samepairs

    fp = pos - tp
    tn = neg - fn

    print "                    Contingency Table"
    print "------------------------------------------------------------"
    print "|                      Same cluster       Diff cluster     |"
    print "|     Same class          TP = %d           FN = %d        |" % (tp, fn)
    print "|     Diff class          FP = %d           TN = %d        |" % (fp, tn)
    print "------------------------------------------------------------"

    # compute Rand index
    return float(tp + tn) / (pos + neg), tp, fp, fn


def f1(clusters, labels, k):
    ''' 
    Evaluate the f1-measure, which more strongly weights false negatives
    than the Rand index.
    '''

    ri, tp, fp, fn = randindex(clusters, labels, k)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def construct(abstracts, vectors, mode):
    ''' 
    Construct a list of vectors for clustering out of the sparse
    representations in abstracts.
    '''
    total = abstracts[0].Get(mode + 'num')
    
    # create vectors for each abstract
    for abstract in abstracts:
        vector = np.zeros(total, dtype=np.float64)  
        text = abstract.Get(mode)
        for i, v in text.iteritems():
            vector[v[0]] = v[1]
        vectors.append(vector)
    return

def label(abstracts, labels):
    ''' 
    Construct a list of labels for the abstracts; this is our 
    ground truth.
    '''
    alllabels = {}
    count = 0       # use this to label the labels

    # go through all the abstract labels
    for abstract in abstracts:
        l = abstract.Get('tags')
        if l not in alllabels:
            alllabels[l] = count
            labels.append(count)    # store the new label id
            count += 1
        else:
            labels.append(alllabels[l]) # store the stored id
    return count

def cluster(abstracts, mode, metric, repeats):
    ''' 
    K-means clustering with evaluation metrics, using custom distance
    function and provided abstracts.
    '''

    # PREPARATION CODE
    metric = euclidean
    repeats = 10

    labels = []
    vectors = []

    # create vectors and labels; k will be number of ground-truth labels
    construct(abstracts, vectors, mode)
    k = label(abstracts, labels)

    # cluster
    clusterer = KMeansClusterer(k, metric, repeats=repeats, 
                                normalise=True)
    clusters = clusterer.cluster(vectors, True, trace=True)

    # compute evaluation metrics
    dist = sumdistance(vectors, clusters, means)
    pure = purity(clusters, labels, k)
    entr = entropy(clusters, labels, k)
    rand, w, w, w = randindex(clusters, labels, k)
    f = f1(clusters, labels, k)

    print "SUM of DISTANCES: %f" % dist
    print "PURITY: %f" % pure
    print "ENTROPY: %f" % entr
    print "RAND INDEX: %f" % rand
    print "F1 MEASURE: %f" % f




