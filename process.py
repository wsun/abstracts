# CS205 Final Project
# Janet Song and Will Sun
#
# Process abstracts for similarity analysis.
# 
# Bag-of-words and bigram representations, also stopword removal and tf-idf
# calculation

import re
import sys
from sys import argv
import csv
from collections import defaultdict
import numpy as np
import scipy as sp
import math
from mpi4py import MPI

import similar as Similar
from abstract import Abstract

# Serial load
def load(filename, abstracts, dictionary, stops):
    dictlist = []
    absids = []
            
    with open(filename) as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in absids:
                abs, dictlist = load_abs(row, dictlist, stops)
                abstracts.append(abs)
                absids.append(row[0])

    # create dictionary
    dictionary = create_dict(dictlist, dictionary)

# load abstract
def load_abs(row, dictlist, stops):
    abs = Abstract(row[0])
    abs.Set('title', row[1])
    abs.Set('text', row[2][10:])
    abs.Set('tags', row[3].split(','))
    
    # remove stop words and clean text
    abstext = [''.join([c.lower() for c in word if c.isalnum()]) for word in row[2][10:].split() if word not in stops]
    abs.Set('cleantext', abstext)
    
    for word in abstext:
        dictlist.append(word)
    
    return abs, dictlist
    
# Create dictionary
def create_dict(dictlist, dictionary):
    dictlist = [word for word in dictlist if dictlist.count(word) > 1]
    for word in dictlist:
        if word not in dictionary:
            dictionary.append(word)
    dictionary.sort()
    return dictionary

# Serial bag of words
def create_bagofwords(abstract, dictionary):
    bow = defaultdict(float)
    abstext = abstract.Get('cleantext')
    for word in abstext:
        if word in dictionary:
            ind = dictionary.index(word)
            bow[ind] += 1.0
    normalize(bow)
    return bow

# Serial bigrams
def create_bigram(abstract, dictionary, bigramdict):
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

# Serial TFIDF for bag of words or bigrams
def serial_tfidf(abstracts, type, termdoc, ex=None):
    numabs = float(len(abstracts))
    for abstract in abstracts:
        tfidf = create_tfidf(abstract, termdoc, numabs, type)
        abstract.Set('tfidf'+type, tfidf)
        if ex:
            abstract.Set('bigramnum', ex)

# Find TFIDF for type
def create_tfidf(abstract, termdoc, numabs, type):
    tfidf = defaultdict(float)
    for ind, freq in abstract.Get(type).iteritems():
        tfidf[ind] = freq*math.log(numabs/termdoc[ind])
    return tfidf

# Find number of documents in which a phrase or word appears
def termall(abstracts, type):
    termall = defaultdict(float)
    for abstract in abstracts:
        for ind, count in abstract.Get(type).iteritems():
            termall[ind] += 1.0
    return termall

# Serial normalize
def normalize(array):
    numwords = float(sum(array.values()))
    for ind, count in array.iteritems():
        array[ind] = count/numwords
    return array

# Master for MPI
def master(comm, filename):
    # initialize variables
    size = comm.Get_size()
    status = MPI.Status()
    abstracts = []
    dictlist = []
    dictionary = []
    termbow = defaultdict(float)
    termbigram = defaultdict(float)
    numabs = 0

    # load stop words
    stops = set()
    stop_file = 'stopwords.txt'
    with open(stop_file, 'rU') as stopFile:
        for row in stopFile.readlines():
            stops.add(row.replace('\n', ''))

    initial = 1
    # Load abstracts
    absids = []
    print "Loading abstracts ..."
    with open(filename) as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in absids:
                absids.append(row[0])
                # send first row to each slave
                # TODO: check if size > num of rows
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
    print abstracts
    
    # Create dictionary
    print "Creating dictionary ..."
    dictionary = create_dict(dictlist, dictionary)
    dictlength = len(dictionary)
    #print dictionary
    
    # Bag of words and Bigrams
    print "Creating bag of words and bigrams ..."
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
            for ind in bow.keys():
                termbow[ind] += 1.0
            for ind in bigram.keys():
                termbigram[ind] += 1.0
            comm.send((abstract, dictionary), dest=status.Get_source(), tag=ind)  
            ind += 1
    
    # tell slaves when there are no abstracts left
    for rank in range(1,size):
        bow, bigram, bigrampartdict = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        abstracts[status.Get_tag()].Set('bow', bow)
        abstracts[status.Get_tag()].Set('bownum', dictlength)
        abstracts[status.Get_tag()].Set('bigram', bigram)
        bigramdict.extend([tup for tup in bigrampartdict if tup not in bigramdict])
        for ind in bow.keys():
            termbow[ind] += 1.0
        for ind in bigram.keys():
            termbigram[ind] += 1.0
        comm.send((None, None), dest=status.Get_source(), tag=ind)
    bigramdictlen = len(bigramdict)
    
    # Find number of documents in which terms appear in all documents (for TF-IDF)
    print "Finding term frequency ..."
    #termbow = termall(abstracts, 'bow')
    #termbigram = termall(abstracts, 'bigram')
    numabs = float(len(abstracts))

    # TF-IDF
    print "Creating TF-IDF ..."
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
    
    print "Done!"        
    return abstracts, dictionary

# Slave for MPI
def slave(comm):
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
        abs, dictlist = load_abs(row, dictlist, stops)
        
        # send abstract back to master
        comm.send((abs, dictlist), dest=0)

    # Find bag of words and bigram
    print "Slave: find bow and bigram"
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
    print "Slave: TF-IDF"
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
    
    return

def main_parallel(comm, filename):
    # Get MPI data
    rank = comm.Get_rank()
    abstracts = []

    # Load and process data
    if rank == 0:
        abstracts, dictionary = master(comm, filename)
    else:    
        slave(comm)
    
    #if rank == 0:
    #    for abstract in abstracts:
    #        print abstract.Get('bownum')
    #        print abstract.Get('bigramnum')
    #        print abstract.Get('tfidfbigram')

    return abstracts
    
# Find similarity matrices
def main_parallel_sim(comm, absind, abstracts, type, mattype):
    rank = comm.Get_rank()
    if rank == 0:
        print "Parallel version: Similarity matrices"
        cossim_matrix, jaccard_matrix = Similar.master(comm, absind, abstracts, type)
        if mattype == 'cossim':
            return cossim_matrix
        else:
            return jaccard_matrix
    else:
        Similar.slave(comm)

def main_serial(comm, filename):
    rank = comm.Get_rank()
    # serial version
    if rank == 0:
        print "Serial version ..."
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
            for pair in bigramdict.keys():
                termbigram[pair] += 1.0
        # create dict of tfidf
        serial_tfidf(abstracts, 'bow', termbow, len(bigramdict))
        serial_tfidf(abstracts, termbigram, 'bigram')

# Find similarity matrices
def main_serial_sim(comm, absind, abstracts, type, mattype):
    rank = comm.Get_rank()
    if rank == 0:
        # Similarity matrices
        print "Serial version: Similarity matrices"
        cossim_matrix, jaccard_matrix = Similar.calculate_similarity_matrices(absind, abstracts, type)
        if mattype == 'cossim':
            return cossim_matrix
        else:
            return jaccard_matrix

if __name__ == '__main__':
    script, filename, version = argv
    
    if version.lower() == 'p':
        main_parallel(filename)
    else:
        main_serial(filename)


