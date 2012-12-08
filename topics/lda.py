# CS205 Final Project
# Janet Song and Will Sun
#
# LDA with MPI master/slave

import ldamodel as custom
from gensim import corpora, models, similarities
import os, logging, time, sys
from mpi4py import MPI

# debug
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# tags for worker action
DIE = 0
RESET = 1
MERGE = 2
WORK = 3

def slave(comm, dictionary, num_topics, chunksize, passes, updates, alpha, eta, decay):
    rank = comm.Get_rank()
    logger = logging.getLogger('gensim.models.lsi_worker')
    jobsdone = 0

    logger.info("initializing worker #%s" % rank)
    model = custom.LdaModel(num_topics=num_topics, 
                            id2word=dictionary, chunksize=chunksize,
                            passes=passes, update_every=updates,
                            alpha=alpha, eta=eta,
                            distributed=False)

    # wait around for jobs, process them as they come in
    status = MPI.Status()
    while True:
        job = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        # time to die
        if tag == DIE:
            break

        # prepare for a new EM iteration
        elif tag == RESET:
            assert job is not None
            logger.info("resetting worker #%i" % rank)
            model.state = job
            model.sync_state()
            model.state.reset()
            comm.send(None, dest=0) # let master know we're finished

        # send master some results
        elif tag == MERGE:
            logger.info("worker #%i returning its state after %s jobs" %
                    (rank, jobsdone))
            result = model.state
            assert isinstance(result, custom.LdaState)
            model.clear() # free up mem in-between two EM cycles
            comm.send(result, dest=0)

        # master wants some work done
        elif tag == WORK:
            model.do_estep(job)
            jobsdone += 1
            comm.send(None, dest=0)

    # goodbye
    logger.info("terminating worker #%i" % rank)
    return

def master(comm, corpus, dictionary, num_topics, chunksize, passes, updates, alpha, eta, decay):
    model = custom.LdaModel(corpus=corpus, num_topics=num_topics, 
                            id2word=dictionary, chunksize=chunksize,
                            passes=passes, update_every=updates,
                            alpha=alpha, eta=eta,
                            distributed=True, comm=comm)
    return model

def serial(corpus, dictionary, num_topics, chunksize, passes, updates, alpha, eta, decay):
    model = custom.LdaModel(corpus=corpus, num_topics=num_topics, 
                            id2word=dictionary, chunksize=chunksize,
                            passes=passes, update_every=updates,
                            alpha=alpha, eta=eta,
                            distributed=False)
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
    num_topics = 15
    chunksize = 100
    alpha = None
    eta = None
    updates = 1
    passes = 100
        
    if rank == 0:
        p_start = MPI.Wtime()
        p_model = master(comm, corpus, dictionary, num_topics, chunksize, passes, updates, alpha, eta, decay)
        p_stop = MPI.Wtime()

        # check
        if debug:
            s_start = time.time()
            s_model = serial(corpus, dictionary, num_topics, chunksize, passes, updates, alpha, eta, decay)
            s_stop = time.time()

            print "PARALLEL MODEL:"
            pretty(p_model.show_topics(num_topics=num_topics))
            print "SERIAL MODEL:"
            pretty(s_model.show_topics(num_topics=num_topics))

            print "Serial Time: %f secs" % (s_stop - s_start)
        print "Parallel Time: %f secs" % (p_stop - p_start)

    else:
        slave(comm, dictionary, num_topics, chunksize, passes, updates, alpha, eta, decay)
