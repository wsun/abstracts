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
    script, filename = sys.argv

    ploadstart = 0.0
    if rank == 0:
        ploadstart = MPI.Wtime()
    abstracts = Process.main_parallel(comm, filename)
    if rank == 0:
        ploadend = MPI.Wtime()
        print "Parallel load time: %f secs" % (ploadend - ploadstart)
    
    psimstart = 0.0
    if rank == 0:
        psimstart = MPI.Wtime()
    Process.main_parallel_sim(comm, 2, abstracts, 'bow', 'cossim')
    if rank == 0:
        psimend = MPI.Wtime()
        print "Parallel similarity time: %f secs" % (psimend - psimstart)

    # Serial version
    if rank == 0:
        sloadstart = time.time()
        abstracts = Process.main_serial(comm, filename)
        sloadend = time.time()
        print "Serial load time: %f secs" % (sloadend - sloadstart)

        ssimstart = time.time()
        matrix = Process.main_serial_sim(comm, 2, abstracts, 'bow', 'cossim')
        ssimend = time.time()
        print "Serial similarity time: %f secs" % (ssimend - ssimstart)

    # More parallel testing
    if rank == 0:
        print "Parallel testing ... "
        dictionary = []
        numabs = 0

        # load stop words
        stops = set()
        stop_file = 'stopwords.txt'
        with open(stop_file, 'rU') as stopFile:
            for row in stopFile.readlines():
                stops.add(row.replace('\n', ''))

        ploadstart = MPI.Wtime()
        abstracts, dictlist = Process.master_load(comm, filename, stops)

        # Create dictionary
        #print "Creating dictionary ..."
        Process.create_dict(dictlist, dictionary)
        #print dictionary
        ploadend = MPI.Wtime()

        # Find bow and bigram
        pfreqstart = MPI.Wtime()
        bigramdictlen, termbow, termbigram = Process.master_bowbigram(comm, abstracts, dictionary)
        pfreqend = MPI.Wtime()

        # Find tfidf
        ptfidfstart = MPI.Wtime()
        Process.master_tfidf(comm, abstracts, dictionary, bigramdictlen, termbow, termbigram)
        ptfidfend = MPI.Wtime()

        # print times
        print "Parallel times"
        print "Load time: %f secs" % (ploadend - ploadstart)
        print "Load time: %f secs" % (pfreqend - pfreqstart)
        print "Load time: %f secs" % (ptfidfend - ptfidfstart)
    else:    
        Process.slave(comm)


    # More serial testing
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

        sloadstart = time.time()
        Process.load(filename, abstracts, dictionary, stops) 
        sloadend = time.time()

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

        # print times
        print "Serial times"
        print "Load time: %f secs" % (sloadend - sloadstart)
        print "Load time: %f secs" % (sfreqend - sfreqstart)
        print "Load time: %f secs" % (stfidfend - stfidfstart)

