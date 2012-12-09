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
import math
from mpi4py import MPI

import similar as Similar
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
                abs, dictlist = load_abs(row, dictlist, stops)
                abstracts.append(abs)
                absids.append(row[0])

    # create dictionary
    dictionary = create_dict(dictlist, dictionary)

def load_abs(row, dictlist, stops):
    '''
    Load an abstract and create an Abstract object.
    Remove stopwords from the abstract to get "cleantext".
    Add to dictlist of all words that appear over all documents.
    '''
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
    return dictionary

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
    return array

def master(comm, filename):
    '''
    Master function MPI implementation for loading and processing abstracts
    '''
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
    #print "Loading abstracts ..."
    with open(filename, "rU") as csvfile:
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
    #print abstracts
    
    # Create dictionary
    #print "Creating dictionary ..."
    dictionary = create_dict(dictlist, dictionary)
    dictlength = len(dictionary)
    #print dictionary
    
    # Bag of words and Bigrams
    #print "Creating bag of words and bigrams ..."
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
    bigramdictlen = len(bigramdict)
    
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
    
    #print "Done!"        
    return abstracts, dictionary


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
        abs, dictlist = load_abs(row, dictlist, stops)
        
        # send abstract back to master
        comm.send((abs, dictlist), dest=0)

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
        cossim, jaccard = Similar.master(comm, absind, abstracts, type)
        if mattype == 'cossim':
            #print cossim
            return cossim
        else:
            #print jaccard
            return jaccard
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
        cossim, jaccard = Similar.calculate_similarity_matrices(absind, abstracts, type)
        if mattype == 'cossim':
            #print cossim
            return cossim
        else:
            #print jaccard
            return jaccard

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
    # Serial version
    elif version.lower() == 's':
        abstracts = main_serial(comm, filename)
        #matrix = main_serial_sim(comm, 2, abstracts, 'bow', 'cossim')
        


