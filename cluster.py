# CS205 Final Project
# Janet Song and Will Sun
#
# K-means clustering, using NLTK
# 
# Evaluate different representations and similarity measures

from nltk.cluster import KMeansClusterer
from collections import defaultdict
from gensim import corpora, models
import process as Process
import lsa as Lsa
import lda as Lda
import numpy as np
import math, sys

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
    print "|                      Same cluster       Diff cluster"
    print "|     Same class         TP = %d           FN = %d" % (tp, fn)
    print "|     Diff class         FP = %d           TN = %d" % (fp, tn)
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
    return f1, ri


def construct(abstracts, vectors, mode):
    ''' 
    Construct a list of vectors for clustering out of the sparse
    representations in abstracts.
    '''
    total = None
    if mode == 'lsa' or mode == 'lda':
        total = abstracts[0].Get('numtopics')
    else:
        total = abstracts[0].Get(mode + 'num')

    # force usage of tfidf-transformed vectors
    if mode == 'bow' or mode == 'bigram':
        mode = 'tfidf' + mode
    
    # create vectors for each abstract
    for abstract in abstracts:
        vector = np.zeros(total, dtype=np.float64)  
        text = abstract.Get(mode)
        for k, v in text.iteritems():
            vector[k] = v
        vectors.append(vector)
    return

def label(abstracts, labels):
    ''' 
    Construct a list of labels for the abstracts; this is our 
    ground truth.
    '''
    alllabels = {}
    count = 0       # use this to label the labels

    # go through all the abstract labels, assume single label
    for abstract in abstracts:
        l = abstract.Get('tags')[0]
        if l not in alllabels:
            alllabels[l] = count
            labels.append(count)    # store the new label id
            count += 1
        else:
            labels.append(alllabels[l]) # store the stored id
    return count

def process(filename):
    ''' Serial processing of abstracts, for evaluation purposes. '''
    abstracts = []
    dictionary = []

    # load stop words
    stops = set()
    stop_file = 'stopwords.txt'
    with open(stop_file, 'rU') as stopFile:
        for row in stopFile.readlines():
            stops.add(row.replace('\n', ''))
    
    Process.load(filename, abstracts, dictionary, stops) 
    dictlength = len(dictionary) 
    bigramdict = []
    termbow = defaultdict(float)
    termbigram = defaultdict(float)
    for abstract in abstracts:
        # create dict of word frequency (bag of words)
        bow = Process.create_bagofwords(abstract, dictionary)
        abstract.Set('bow', bow)
        abstract.Set('bownum', dictlength)
        for ind in bow.keys():
            termbow[ind] += 1.0
        # create dict of bigram frequency
        bigram, bigramdict = Process.create_bigram(abstract, dictionary, bigramdict)
        abstract.Set('bigram', bigram)
        for pair in bigram.keys():
            termbigram[pair] += 1.0
    # create dict of tfidf
    Process.serial_tfidf(abstracts, 'bow', termbow, len(bigramdict))
    Process.serial_tfidf(abstracts, 'bigram', termbigram)

    ##### TOPICS
    # prepare dictionary and corpora for topic modeling
    docs = [abstract.Get('cleantext') for abstract in abstracts]
    dictionary = corpora.Dictionary(docs)
    dictionary.save('abstracts.dict')           
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize('abstracts.mm', corpus)

    # use gensim tfidf to transform
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # load lsa and lda models
    numtopics = 15  # this can be adjusted
    lsaModel = Lsa.serial(corpus_tfidf, dictionary, numtopics)
    ldaModel = Lda.serial(corpus_tfidf, dictionary, numtopics)

    # store lda and lsa representation in all abstracts
    for i in xrange(len(abstracts)):
        lsaVec = lsaModel[tfidf[corpus[i]]]
        ldaVec = ldaModel[tfidf[corpus[i]]]
        lsaVector = defaultdict(float)
        ldaVector = defaultdict(float)
        for v in lsaVec:
            lsaVector[v[0]] = v[1]
        for v in ldaVec:
            ldaVector[v[0]] = v[1]
        abstracts[i].Set('lsa', lsaVector)
        abstracts[i].Set('lda', ldaVector)
        abstracts[i].Set('numtopics', numtopics)

    return abstracts


def cluster(abstracts, mode, metric, debug=False, repeats=10):
    ''' 
    K-means clustering with evaluation metrics, using custom distance
    function and provided abstracts.
    '''

    labels = []
    vectors = []

    # create vectors and labels; k will be number of ground-truth labels
    construct(abstracts, vectors, mode)
    k = label(abstracts, labels)

    # cluster
    clusterer = KMeansClusterer(k, metric, repeats=repeats, 
                                normalise=True)
    clusters = clusterer.cluster(vectors, assign_clusters=True, trace=debug) 
    means = clusterer.means()

    print 
    print "EVALUATION:"

    # compute evaluation metrics
    dist = sumdistance(vectors, clusters, means)
    pure = purity(clusters, labels, k)
    entr = entropy(clusters, labels, k)
    f, rand = f1(clusters, labels, k)

    print "Sum of distances: %f" % dist
    print "Purity: %f" % pure
    print "Entropy: %f" % entr
    print "Rand index: %f" % rand
    print "F1 measure: %f" % f

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: ' + sys.argv[0] + ' filename [bow, bigram, lsa, lda] [euc, cos, jac]'
        sys.exit(0)

    filename = sys.argv[1]
    if sys.argv[2] != 'bow' and sys.argv[2] != 'bigram' and sys.argv[2] != 'lsa' and sys.argv[2] != 'lda':
        print 'Please specify a matrix representation: bow, bigram, lsa, lda'
        sys.exit(0)
    mode = sys.argv[2]
    
    metric = None
    if sys.argv[3] == 'euc':
        metric = euclidean
    elif sys.argv[3] == 'cos':
        metric = cosine
    elif sys.argv[3] == 'jac':
        metric = jaccard
    else:
        print 'Please specify a distance metric to examine: euc(lidean), cos(ine), jac(card)'
        sys.exit(0)

    print 'Loading...'
    abstracts = process(filename)
    print 'Clustering...'
    cluster(abstracts, mode, metric) # set debug=True for trace
