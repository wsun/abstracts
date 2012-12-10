# CS205 Final Project
# Janet Song and Will Sun
#
# LSA with MPI master/slave

import lsimodel as custom
from gensim import corpora, models, similarities
import os, logging, time, sys
from mpi4py import MPI

# debug
#logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

def slave(comm, dictionary, num_topics=15, chunksize=1000, decay=1.0):
    rank = comm.Get_rank()
    logger = logging.getLogger('gensim.models.lsi_worker')
    jobsdone = 0

    logger.info("initializing worker #%s" % rank)
    model = custom.LsiModel(id2word=dictionary, num_topics=num_topics,
                            chunksize=chunksize, decay=decay, 
                            distributed=False)
    # wait around for jobs, process them as they come in 
    while True:
        job = comm.recv(source=0)
        jobsdone += 1

        # receive the kill tag, return all results back to master
        if (job == None):
            logger.info("worker #%i returning its state after %s jobs" %
                (rank, jobsdone))
            assert isinstance(model.projection, custom.Projection)
            result = model.projection
            model.projection = model.projection.empty_like()
            comm.send(result, dest=0)
            break

        # add new job to current model
        else:
            model.add_documents(job)
            comm.send(None, dest=0)

    logger.info("terminating worker #%i" % rank)
    return

def master(comm, corpus, dictionary, num_topics=15, chunksize=1000, decay=1.0):
    size = comm.Get_size()
    status = MPI.Status()
    model = custom.LsiModel(corpus=corpus, num_topics=num_topics, 
                            id2word=dictionary, chunksize=chunksize,
                            decay=decay, distributed=True, comm=comm)
    return model

def serial(corpus, dictionary, num_topics=15, chunksize=1000, decay=1.0):
    
    model = custom.LsiModel(corpus=corpus, num_topics=num_topics,
                            id2word=dictionary, chunksize=chunksize,
                            decay=decay, distributed=False)
    return model

def pretty(topics):
    for t in topics:
        print t

if __name__ == '__main__':

    # get MPI data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        if len(sys.argv) != 1 and len(sys.argv) != 2:
            print 'Usage: ' + sys.argv[0] + ' [--debug]'
            sys.exit(0)

        debug = None
        if len(sys.argv) == 2:
            debug = sys.argv[1]

    # PREPARATION CODE: corpus, dictionary
    dictionary = corpora.Dictionary.load('boom.dict')
    corpus = corpora.MmCorpus('boom.mm')
    num_topics = 100
    chunksize = 500
    decay = 1.0
        
    if rank == 0:
        p_start = MPI.Wtime()
        p_model = master(comm, corpus, dictionary, num_topics, chunksize, decay)
        p_stop = MPI.Wtime()

        # check
        if debug:
            s_start = time.time()
            s_model = serial(corpus, dictionary, num_topics, chunksize, decay)
            s_stop = time.time()

            print "PARALLEL MODEL:"
            pretty(p_model.show_topics(num_topics=num_topics))
            print "SERIAL MODEL:"
            pretty(s_model.show_topics(num_topics=num_topics))

            print "Serial Time: %f secs" % (s_stop - s_start)
            print "Parallel Time: %f secs" % (p_stop - p_start)

    else:
        slave(comm, dictionary, num_topics, chunksize, decay)
