import os
import sys
from sys import argv
import numpy as np
from abstract import Abstract
import process as Process
from mpi4py import MPI
import time
from collections import defaultdict

if __name__ == '__main__':
    # MPI values
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    script, filename = sys.argv

    # Parallel testing
    if rank == 0:
        print size, filename
        print "Parallel testing ... "

        # initialize variables
        dictionary = []

        # load stop words
        stops = set()
        stop_file = 'stopwords.txt'
        with open(stop_file, 'rU') as stopFile:
            for row in stopFile.readlines():
                stops.add(row.replace('\n', ''))
        for r in range(1,size):
            comm.send(stops, dest=r)

        ploadstart = MPI.Wtime()
        abstracts, dictlist = Process.master_load(comm, filename)
        ploadend = MPI.Wtime()

        pdictstart = MPI.Wtime()
        # Create dictionary
        #print "Creating dictionary ..."
        Process.create_dict(dictlist, dictionary)
        # send dictionary everywhere
        for r in range(1,size):
            comm.send(dictionary, dest=r)
        pdictend = MPI.Wtime()

        pcleanstart = MPI.Wtime()
        Process.master_cleantext(comm, abstracts)
        pcleanend = MPI.Wtime()

        pabsstart = MPI.Wtime()
        # send abstracts to all slaves
        for r in range(1,size):
            comm.send(abstracts, dest=r)
        pabsend = MPI.Wtime()


        # Find bow and bigram
        pfreqstart = MPI.Wtime()
        bigramdictlen, termbow, termbigram = Process.master_bowbigram(comm, abstracts, len(dictionary))
        pfreqend = MPI.Wtime()

        psendstart = MPI.Wtime()
        for r in range(1,size):
            comm.send((abstracts, termbow, termbigram), dest=r)
        psendend = MPI.Wtime()

        # Find tfidf
        ptfidfstart = MPI.Wtime()
        Process.master_tfidf(comm, abstracts, bigramdictlen)
        ptfidfend = MPI.Wtime()

        # topic modeling
        ptopicstart = MPI.Wtime()
        Process.master_topics(comm, abstracts)
        ptopicend = MPI.Wtime()

        # print times
        print "Parallel times"
        print "Load time: %f secs" % (ploadend - ploadstart)
        print "Create and send dictionary time: %f secs" % (pdictend - pdictstart)
        print "Clean text time: %f secs" % (pcleanend - pcleanstart)
        print "Send abstract time: %f secs" % (pabsend - pabsstart)
        print "Frequency time: %f secs" % (pfreqend - pfreqstart)
        print "Send abs, terms time: %f secs" % (psendend - psendstart)
        print "TF-IDF time: %f secs" % (ptfidfend - ptfidfstart)
        print "Topic modelling time: %f secs" % (ptopicend - ptopicstart)
    else:    
        Process.slave(comm)

    # Serial testing
    if rank == 0:
        print "Serial testing ..."
        abstracts = []
        dictionary = []

        # load stop words
        stops = set()
        stop_file = 'stopwords.txt'
        with open(stop_file, 'rU') as stopFile:
            for row in stopFile.readlines():
                stops.add(row.replace('\n', ''))

        sloadstart = time.time()
        dictlist = Process.load(filename, abstracts, stops) 
        sloadend = time.time()

        # create dictionary
        sdictstart = time.time()
        Process.create_dict(dictlist, dictionary)
        sdictend = time.time()

        # clean text of words not in dictionary
        scleanstart = time.time()
        for abstract in abstracts:
            abstext = [word for word in abstract.Get('cleantext') if word in dictionary]
            abstract.Set('cleantext', abstext)
        scleanend = time.time()

        sfreqstart = time.time()
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
        sfreqend = time.time()

        # create dict of tfidf
        stfidfstart = time.time()
        Process.serial_tfidf(abstracts, 'bow', termbow, len(bigramdict))
        Process.serial_tfidf(abstracts, 'bigram', termbigram)
        stfidfend = time.time()

        # do some topic modeling
        stopicstart = time.time()
        Process.serial_topics(abstracts)
        stopicend = time.time()

        # print times
        print "Serial times"
        print "Load time: %f secs" % (sloadend - sloadstart)
        print "Create dictionary time: %f secs" % (sdictend - sdictstart)
        print "Clean text time: %f secs" % (scleanend - scleanstart)
        print "Frequency time: %f secs" % (sfreqend - sfreqstart)
        print "TF-IDF time: %f secs" % (stfidfend - stfidfstart)
        print "Topic modelling time: %f secs" % (stopicend - stopicstart)

