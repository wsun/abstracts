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
#from mpi4py import MPI

from abstract import Abstract

# define global variables
dictionary = []
stops = set()
termbow = []
termbigram = []
numabs = 0

# Serial load
def load(filename, abstracts):
    dictlist = []
            
    with open(filename) as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in abstracts:
                abs, dictlist = load_abs(row, dictlist)
                abstracts.append(abs)

    # create dictionary
    create_dict(dictlist)

# load abstract
def load_abs(row, dictlist):
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
def create_dict(dictlist):
    dictlist = [word for word in dictlist if dictlist.count(word) > 1]
    for word in dictlist:
        if word not in dictionary:
            dictionary.append(word)
    dictionary.sort()

# Serial bag of words
def create_bagofwords(abstract):
    bow = defaultdict(float)
    abstext = abstract.Get('cleantext')
    for word in abstext:
        if word in dictionary:
            ind = dictionary.index(word)
            bow[ind] += 1.0
    normalize(bow)
    return bow

# Serial bigrams
def create_bigram(abstract):
    bigram = defaultdict(float)
    abstext = abstract.Get('cleantext')
    for i in range(len(abstext)-1):
        wordgram = abstext[i:i+2]
        wordgram.sort()
        if wordgram[0] in dictionary:
            if wordgram[1] in dictionary:
                pair = (dictionary.index(wordgram[0]),dictionary.index(wordgram[1]))
                bigram[pair] += 1.0
    normalize(bigram)
    return bigram

# Serial TFIDF for bag of words or bigrams
def serial_tfidf(abstracts, type):
    termdoc = termall(abstracts, type)
    numabs = float(len(abstracts))
    for abstract in abstracts:
        tfidf = create_tfidf(abstract, termdoc, numabs, type)
        abstract.Set('tfidf'+type, tfidf)

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

    # Load abstracts
    with open(filename) as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in abstracts:
                # send first row to each slave
                # TODO: check if size > num of rows
                for rank in range(1,size):
                    comm.send(row, dest=rank)
            
                # continue sending rows to slaves
                abs, dict = comm.recv(source=MPI.ANY_SOURCE, status=status)
                abstracts[abs.Get('path')].append(abs)
                dictlist.extend(dict)
                comm.send(row, dest=status.Get_source())
                
        # tell slaves when there are no rows left
        for rank in range(1,size):
            abs, dict = comm.recv(source=MPI.ANY_SOURCE, status=status)
            abstracts[abs.Get('path')].append(abs)
            dictlist.extend(dict)
            comm.send(None, dest=status.Get_source())
    
    # Create dictionary
    create_dict(dictlist)
    
    # Bag of words and Bigrams
    for abstract in abstracts:
        # send first abstract to each slave
        for rank in range(1,size):
            comm.send(abstract, dest=rank)
        
        # continue sending rows to slaves
        bow, bigram = comm.recv(source=MPI.ANY_SOURCE, status=status)
        abstract.Set('bow', bow)
        abstract.Set('bigram', bigram)
        comm.send(abstract, dest=status.Get_source())
        
        # tell slaves when there are no abstracts left
        for rank in range(1,size):
            bow, bigram = comm.recv(source=MPI.ANY_SOURCE, status=status)
            abstract.Set('bow', bow)
            abstract.Set('bigram', bigram)
            comm.send(None, dest=status.Get_source())
    
    # Find number of documents in which terms appear in all documents (for TF-IDF)
    termbow = termall(abstracts, 'bow')
    termbigram = termall(abstracts, 'bigram')
    numabs = float(len(abstracts))

    # TF-IDF
    for abstract in abstracts:
        # send first abstract to each slave
        for rank in range(1,size):
            comm.send(abstract, dest=rank)
        
        # continue sending rows to slaves
        tfidfbow, tfidfbigram = comm.recv(source=MPI.ANY_SOURCE, status=status)
        abstract.Set('tfidfbow', tfidfbow)
        abstract.Set('tfidfbigram', tfidfbigram)
        comm.send(abstract, dest=status.Get_source())
        
        # tell slaves when there are no abstracts left
        for rank in range(1,size):
            tfidfbow, tfidfbigram = comm.recv(source=MPI.ANY_SOURCE, status=status)
            abstract.Set('tfidfbow', tfidfbow)
            abstract.Set('tfidfbigram', tfidfbigram)
            comm.send(None, dest=status.Get_source())
            
    return abstracts

# Slave for MPI
def slave(comm, stops):
    status = MPI.Status()
    
    # Load abstracts
    while True:
        # get message
        row = comm.recv(source=0, status=status)
        
        # end if done
        if not row:
            break
        
        # create Abstract object
        dictlist = []
        abs, dictlist = load_abs(row, dictlist)
        
        # send abstract back to master
        comm.send((abs, dictlist), dest=0)
    
    # Find bag of words and bigram
    while True:
        # get message
        abstract = comm.recv(source=0, status=status)
        
        # end if done
        if not abstract:
            break
        
        # find bag of words
        bow = create_bagofwords(abstract)
        # find bigram
        bigram = create_bigram(abstract)
        
        # send bow and bigram back to master
        comm.send((bow, bigram), dest=0)
    
    # TF-IDF
    while True:
        # get message
        abstract = comm.recv(source=0, status=status)
        
        # end if done
        if not abstract:
            break
        
        # find TF-IDF
        tfidfbow = create_tfidf(abstract, 'bow')
        tfidfbigram = create_tfidf(abstract, 'bigram')
        
        # send bow and bigram back to master
        comm.send((tfidfbow, tfidfbigram), dest=0)
    
    return

if __name__ == '__main__':
    # Get MPI data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    script, filename = argv
    abstracts = []
    
    # parallel version
    if rank == 0:
        # load stop words
        stop_file = 'stopwords.txt'
        with open(stop_file, 'rU') as stopFile:
            for row in stopFile.readlines():
                stops.add(row.replace('\n', ''))
                
        abstracts = master(comm, filename)
    else:    
        slave(comm, stops)
        
    if rank == 0:
        for abstract in abstracts:
            print abstract.Get('tfidfbow')
            print abstract.Get('bigram')
    
    # serial version
    if rank == 0:
        script, filename = argv
        load(filename, abstracts)   
        for abstract in abstracts:
            # create dict of word frequency (bag of words)
            bow = create_bagofwords(abstracts)
            abstract.Set('bow', bow)
            # create dict of bigram frequency
            bigram = create_bigram(abstracts)
            abstract.Set('bigram', bigram)
        # create dict of tfidf
        serial_tfidf(abstracts, 'bow')
        serial_tfidf(abstracts, 'bigram')
        #for abstract in abstracts:
        #    print abstract.Get('tfidfbow')
        #    print abstract.Get('bigram')