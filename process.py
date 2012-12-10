# CS205 Final Project
# Janet Song and Will Sun
#
# Process abstracts for similarity analysis.
# 
# Bag-of-words and bigram representations, also stopword removal and tf-idf
# calculation

import sys
from sys import argv
import time
import csv
from collections import defaultdict
import numpy as np
import math
from mpi4py import MPI
from gensim import corpora, models

import similar as Similar
import lsa as Lsa 
import lda as Lda
from abstract import Abstract

def load(filename, abstracts, dictionary, stops):
    '''
    Serial implementation of loading all abstracts into program/Abstract objects.
    Create dictionary of all words.
    '''
    dictlist = []
    absids = []
          
    with open(filename, "rU") as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in absids:
                abs = load_abs(row, dictlist, stops)
                abstracts.append(abs)
                absids.append(row[0])

    # create dictionary
    create_dict(dictlist, dictionary)

    # clean text of words not in dictionary
    for abstract in abstracts:
        abstext = [word for word in abstract.Get('cleantext') if word in dictionary]
        abstract.Set('cleantext', abstext)
    

def load_abs(row, dictlist, stops):
    '''
    Load an abstract and create an Abstract object.
    Remove stopwords from the abstract to get "cleantext".
    Add to dictlist of all words that appear over all documents.
    '''
    abs = Abstract(row[0])
    abs.Set('title', row[1])
    abs.Set('text', row[2][10:])
    abs.Set('tags', row[3].split(';'))
    
    # remove stop words and clean text
    abstext = [''.join([c.lower() for c in word if c.isalnum()]) for word in row[2][10:].split() if word.lower() not in stops]
    abs.Set('cleantext', abstext)
    
    for word in abstext:
        dictlist.append(word)
    
    return abs
    
def create_dict(dictlist, dictionary):
    '''
    Based on a list of all words (including duplicates) in all documents, 
    create list of words (without duplicates). 
    Remove all words that only occur once in all documents.
    '''
    dictlist = [word for word in dictlist if dictlist.count(word) > 1]
    for word in dictlist:
        if word not in dictionary:
            dictionary.append(word)
    dictionary.sort()

def create_bagofwords(abstract, dictionary):
    '''
    Finds bag of word frequencies for an abstract given a dictionary.
    '''
    bow = defaultdict(float)
    abstext = abstract.Get('cleantext')
    for word in abstext:
        if word in dictionary:
            ind = dictionary.index(word)
            bow[ind] += 1.0
    normalize(bow)
    return bow

def create_bigram(abstract, dictionary, bigramdict):
    '''
    Find bigram frequencies for an abstract given a dictionary.
    Adds unique bigrams to the bigramdict to get a count of all bigrams for
    TFIDF implementation.
    '''
    bigram = defaultdict(float)
    abstext = abstract.Get('cleantext')
    for i in range(len(abstext)-1):
        wordgram = abstext[i:i+2]
        wordgram.sort()
        if wordgram[0] in dictionary:
            if wordgram[1] in dictionary:
                pair = (dictionary.index(wordgram[0]),dictionary.index(wordgram[1]))
                bigram[pair] += 1.0
        if wordgram not in bigramdict:
            bigramdict.append(wordgram)
    normalize(bigram)
    return bigram, bigramdict

def serial_tfidf(abstracts, type, termdoc, ex=None):
    '''
    Serial implementation of TFIDF for bag of words or bigrams (type)
    '''
    numabs = float(len(abstracts))
    for abstract in abstracts:
        tfidf = create_tfidf(abstract, termdoc, numabs, type)
        abstract.Set('tfidf'+type, tfidf)
        # add overall number of bigrams to each abstract object
        if ex:
            abstract.Set('bigramnum', ex)

def create_tfidf(abstract, termdoc, numabs, type):
    '''
    Find TFIDF for an abstract for either bag of words or bigram (type)
    '''
    tfidf = defaultdict(float)
    for ind, freq in abstract.Get(type).iteritems():
        tfidf[ind] = freq*math.log(numabs/termdoc[ind])
    return tfidf

def normalize(array):
    '''
    Normalize an array to [0,1]
    '''
    numwords = float(sum(array.values()))
    for ind, count in array.iteritems():
        array[ind] = count/numwords

def serial_topics(abstracts):
    ''' Serial computation of topic models for all abstracts. '''
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


def master(comm, filename):
    '''
    Master function MPI implementation for loading and processing abstracts
    '''
    # initialize variables
    size = comm.Get_size()
    status = MPI.Status()
    dictionary = []

    abstracts, dictlist = master_load(comm, filename)

    # Create dictionary
    #print "Creating dictionary ..."
    create_dict(dictlist, dictionary)
    #print dictionary

    # clean text of words not in dictionary
    #print "Cleaning text ..."
    master_cleantext(comm, abstracts, dictionary)

    # Find bow and bigram
    bigramdictlen, termbow, termbigram = master_bowbigram(comm, abstracts, dictionary)

    # Find tfidf
    master_tfidf(comm, abstracts, dictionary, bigramdictlen, termbow, termbigram)

    return abstracts, dictionary


def master_load(comm, filename):
    '''
    Master function MPI implementation for loading abstracts into Abstract objects
    '''
    size = comm.Get_size()
    status = MPI.Status()

    # load stop words
    stops = set()
    stop_file = 'stopwords.txt'
    with open(stop_file, 'rU') as stopFile:
        for row in stopFile.readlines():
            stops.add(row.replace('\n', ''))

    abstracts = []
    dictlist = []
    initial = 1
    # Load abstracts
    absids = []
    #print "Loading abstracts ..."
    with open(filename, "rU") as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in absids:
                absids.append(row[0])
                # send first row to each slave
                if initial < size:
                    comm.send((row,stops), dest=initial)
                    initial += 1
                else:
                    # continue sending rows to slaves
                    abs, dict = comm.recv(source=MPI.ANY_SOURCE, status=status)
                    abstracts.append(abs)
                    dictlist.extend(dict)
                    comm.send((row, stops), dest=status.Get_source())
                
        # tell slaves when there are no rows left
        for rank in range(1,size):
            abs, dict = comm.recv(source=MPI.ANY_SOURCE, status=status)
            abstracts.append(abs)
            dictlist.extend(dict)
            comm.send((None, None), dest=status.Get_source())
    #print abstracts
    return abstracts, dictlist
    

def master_bowbigram(comm, abstracts, dictionary):
    '''
    Master function MPI implementation for finding bag of words and
    bigrams for abstracts
    '''
    size = comm.Get_size()
    status = MPI.Status()
    # Bag of words and Bigrams
    #print "Creating bag of words and bigrams ..."
    dictlength = len(dictionary)
    termbow = defaultdict(float)
    termbigram = defaultdict(float)
    ind = 0
    bigramdict = []
    for abstract in abstracts:
        # send first abstract to each slave
        if ind < size-1:
            comm.send((abstract, dictionary), dest=ind+1, tag=ind)
            ind += 1
        # continue sending rows to slaves
        else:
            bow, bigram, bigrampartdict = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            abstracts[status.Get_tag()].Set('bow', bow)
            abstracts[status.Get_tag()].Set('bownum', dictlength)
            abstracts[status.Get_tag()].Set('bigram', bigram)
            bigramdict.extend([tup for tup in bigrampartdict if tup not in bigramdict])
            for key in bow.keys():
                termbow[key] += 1.0
            for key in bigram.keys():
                termbigram[key] += 1.0
            comm.send((abstract, dictionary), dest=status.Get_source(), tag=ind)  
            ind += 1
    
    # tell slaves when there are no abstracts left
    for rank in range(1,size):
        bow, bigram, bigrampartdict = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        abstracts[status.Get_tag()].Set('bow', bow)
        abstracts[status.Get_tag()].Set('bownum', dictlength)
        abstracts[status.Get_tag()].Set('bigram', bigram)
        bigramdict.extend([tup for tup in bigrampartdict if tup not in bigramdict])
        for key in bow.keys():
            termbow[key] += 1.0
        for key in bigram.keys():
            termbigram[key] += 1.0
        comm.send((None, None), dest=status.Get_source())
    
    return len(bigramdict), termbow, termbigram

def master_cleantext(comm, abstracts, dictionary):
    '''
    Master function MPI implementation for cleaning text for later LSA/LDA
    '''
    size = comm.Get_size()
    status = MPI.Status()
    ind = 0
    # clean text of words not in dictionary
    for abstract in abstracts:
        # send first abstract to each slave
        if ind < size-1:
            comm.send((abstract.Get('cleantext'), dictionary), dest=ind+1, tag=ind)
            ind += 1
        # continue sending rows to slaves
        else:
            abstext = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            abstracts[status.Get_tag()].Set('cleantext', abstext)
            comm.send((abstract.Get('cleantext'), dictionary), dest=status.Get_source(), tag=ind)
            ind += 1
        
    # tell slaves when there are no abstracts left
    for rank in range(1,size):
        abstext = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        abstracts[status.Get_tag()].Set('cleantext', abstext)
        comm.send((None, None), dest=status.Get_source(), tag=ind)


def master_tfidf(comm, abstracts, dictionary, bigramdictlen, termbow, termbigram):
    '''
    Master function MPI implementation for finding TF-IDF bag of words
    and bigrams for abstracts
    '''
    size = comm.Get_size()
    status = MPI.Status()
    # Find number of documents in which terms appear in all documents (for TF-IDF)
    #print "Finding term frequency ..."
    numabs = float(len(abstracts))

    # TF-IDF
    #print "Creating TF-IDF ..."
    ind = 0
    for abstract in abstracts:
        # send first abstract to each slave
        if ind < size-1:
            comm.send((abstract, termbow, termbigram, numabs), dest=ind+1, tag=ind)
            ind += 1
        # continue sending rows to slaves
        else:
            tfidfbow, tfidfbigram = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            abstracts[status.Get_tag()].Set('tfidfbow', tfidfbow)
            abstracts[status.Get_tag()].Set('tfidfbigram', tfidfbigram)
            abstracts[status.Get_tag()].Set('bigramnum', bigramdictlen)
            comm.send((abstract, termbow, termbigram, numabs), dest=status.Get_source(), tag=ind)
            ind += 1
        
    # tell slaves when there are no abstracts left
    for rank in range(1,size):
        tfidfbow, tfidfbigram = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        abstracts[status.Get_tag()].Set('tfidfbow', tfidfbow)
        abstracts[status.Get_tag()].Set('tfidfbigram', tfidfbigram)
        abstracts[status.Get_tag()].Set('bigramnum', bigramdictlen)
        comm.send((None, None, None, None), dest=status.Get_source(), tag=ind)

def master_topics(comm, abstracts):
    ''' Master function for distributed topic modeling. '''
    numworkers = comm.Get_size() - 1
    
    # prepare topic models
    docs = [abstract.Get('cleantext') for abstract in abstracts]
    dictionary = corpora.Dictionary(docs)
    dictionary.save('abstracts.dict')           
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpora.MmCorpus.serialize('abstracts.mm', corpus)

    # send init messages
    for i in xrange(numworkers):
        comm.send(42, dest=i+1)

    # wait for ready
    for i in xrange(numworkers):
        comm.recv(source=i+1)

    # tfidf transformation
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # create models in parallel
    numtopics = 15  # this can be adjusted
    lsaModel = Lsa.master(comm, corpus_tfidf, dictionary, numtopics)
    ldaModel = Lda.master(comm, corpus_tfidf, dictionary, numtopics)

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

def slave(comm):
    '''
    Slave function for MPI implementation for loading and processing abstracts
    '''
    status = MPI.Status()
    
    # Load abstracts
    while True:
        # get message
        row, stops = comm.recv(source=0, status=status)
        
        # end if done
        if not row:
            break
        
        # create Abstract object
        dictlist = []
        abs = load_abs(row, dictlist, stops)
        
        # send abstract back to master
        comm.send((abs, dictlist), dest=0)

    # clean abstracts
    while True:
        # get message
        abstext, dictionary = comm.recv(source=0, tag = MPI.ANY_TAG, status=status)
        
        # end if done
        if not abstext:
            break
        
        # clean text
        abstext = [word for word in abstext if word in dictionary]

        # send abstract back to master
        comm.send(abstext, dest=0, tag=status.Get_tag())

    # Find bag of words and bigram
    #print "Slave: find bow and bigram"
    while True:
        # get message
        abstract, dictionary = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        # end if done
        if not abstract:
            break
        
        # find bag of words
        bow = create_bagofwords(abstract, dictionary)
        # find bigram
        bigramdict = []
        bigram, bigramdict = create_bigram(abstract, dictionary, bigramdict)
        
        # send bow and bigram back to master
        comm.send((bow, bigram, bigramdict), dest=0, tag=status.Get_tag())
    
    # TF-IDF
    #print "Slave: TF-IDF"
    while True:
        # get message
        abstract, termbow, termbigram, numabs = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        
        # end if done
        if not abstract:
            break
        
        # find TF-IDF
        tfidfbow = create_tfidf(abstract, termbow, numabs, 'bow')
        tfidfbigram = create_tfidf(abstract, termbigram, numabs, 'bigram')
        
        # send bow and bigram back to master
        comm.send((tfidfbow, tfidfbigram), dest=0, tag=status.Get_tag())

    ##### TOPICS
    # topic modeling init
    dictionary = None
    
    # get message to begin working
    init = comm.recv(source=0)
    if init == 42:
        dictionary = corpora.Dictionary.load('abstracts.dict')
        comm.send(43, dest=0)
    Lsa.slave(comm, dictionary)
    Lda.slave(comm, dictionary)
        
    return

def main_parallel(comm, filename):
    '''
    MPI implementation for loading and processing abstracts
    '''
    # Get MPI data
    rank = comm.Get_rank()
    abstracts = []

    # Load and process data
    if rank == 0:
        abstracts, dictionary = master(comm, filename)
        
    else:    
        slave(comm)

    return abstracts
    
# Find similarity matrices
def main_parallel_sim(comm, absind, abstracts, type, mattype):
    '''
    MPI implementation to find similarity for the mattype (cosine or jaccard 
    distance) between a given abstract (given by id, absind) and all abstracts
    based on their "type" values
    '''
    rank = comm.Get_rank()
    if rank == 0:
        #print "Parallel version: Similarity matrices"
        simvalues = Similar.master(comm, absind, abstracts, type, mattype)
        return simvalues
    else:
        Similar.slave(comm)

def main_serial(comm, filename):
    '''
    Load and process abstracts (serial)
    '''
    rank = comm.Get_rank()
    if rank == 0:
        #print "Serial version ..."
        abstracts = []
        dictionary = []

        # load stop words
        stops = set()
        stop_file = 'stopwords.txt'
        with open(stop_file, 'rU') as stopFile:
            for row in stopFile.readlines():
                stops.add(row.replace('\n', ''))
        
        load(filename, abstracts, dictionary, stops) 
        dictlength = len(dictionary) 
        bigramdict = []
        termbow = defaultdict(float)
        termbigram = defaultdict(float)
        for abstract in abstracts:
            # create dict of word frequency (bag of words)
            bow = create_bagofwords(abstract, dictionary)
            abstract.Set('bow', bow)
            abstract.Set('bownum', dictlength)
            for ind in bow.keys():
                termbow[ind] += 1.0
            # create dict of bigram frequency
            bigram, bigramdict = create_bigram(abstract, dictionary, bigramdict)
            abstract.Set('bigram', bigram)
            for pair in bigram.keys():
                termbigram[pair] += 1.0
        # create dict of tfidf
        serial_tfidf(abstracts, 'bow', termbow, len(bigramdict))
        serial_tfidf(abstracts, 'bigram', termbigram)

        # do some topic modeling
        serial_topics(abstracts)

        #print abstracts[0].Get('bownum')
        #print abstracts[0].Get('bigramnum')
        #for i in range(5):
        #    print abstracts[i*5].Get('bow')
        return abstracts

def main_serial_sim(comm, absind, abstracts, type, mattype):
    '''
    Find similarity (Serial) for the mattype (cosine or jaccard distance) 
    between a given abstract (given by the id, absind) and all abstracts 
    based on their "type" values.
    '''
    rank = comm.Get_rank()
    if rank == 0:
        # Similarity matrices
        #print "Serial version: Similarity matrices"
        simvalues = Similar.calculate_similarity_matrices(absind, abstracts, type, mattype)
        return simvalues

if __name__ == '__main__':
    # MPI values
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # check input
    version = 'p'
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        if rank == 0:
            print 'Usage: ' + sys.argv[0] + ' filename' + ' [p or s]'
            sys.exit(0)
        else:
            sys.exit(0)
        
    if len(sys.argv) == 3:
        version = sys.argv[2]
    filename = sys.argv[1]
    
    # Parallel version
    if version.lower() == 'p':
        abstracts = main_parallel(comm, filename)
        #matrix = main_parallel_sim(comm, 2, abstracts, 'bow', 'cossim')
        #print matrix
    # Serial version
    elif version.lower() == 's':
        if rank == 0:
            abstracts = main_serial(comm, filename)
            #matrix = main_serial_sim(comm, 2, abstracts, 'bow', 'cossim')
            #print matrix


