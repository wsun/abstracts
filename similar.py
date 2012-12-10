# CS205 Final Project
# Janet Song and Will Sun
#
# Calculate similarity between documents.
# 

import abstract as Abstract
import numpy as np
from mpi4py import MPI

def cosine_similarity(abs1, abs2, type):
    '''
    Finds cosine similarity for two given abstracts based on their "type" 
    values, i.e. if type is 'bow', then cosine similarity is determined based 
    on bag of words dictionary
    '''
    text1 = abs1.Get(type)
    text2 = abs2.Get(type)
    num = float(0)
    for ind, count in text1.iteritems():
        if ind in text2:
            num += float(count*text2[ind])
    denom = float(np.linalg.norm(text1.values())*np.linalg.norm(text2.values()))
    return float(num/denom)

def jaccard_index(abs1, abs2, type):
    '''
    Finds jaccard index for two given abstracts based on their "type" values, 
    i.e. if type is 'bow', then jaccard index is determined based on bag of 
    words dictionary
    '''
    text1 = abs1.Get(type)
    text2 = abs2.Get(type)
    sameattr = 0.0
    numofattr = 0.0
    for ind, count in text1.iteritems():
        if ind in text2:
            sameattr += 1.0
        numofattr += 1.0
    numofattr += len(text2) - sameattr
    return float(sameattr/numofattr)
    
def calculate_similarity_matrices(absind, abstracts, type, mattype):
    '''
    Find similarity (Serial) for the cosine distance or jaccard distance
    (matttype) between a given abstract (given by the id, absind) and 
    all abstracts based on their "type" values.
    '''
    simvalues = np.float64(np.zeros(len(abstracts)))
    for i in range(len(abstracts)):
        if mattype == 'cossim':
            simvalues[i] = 1.0 - cosine_similarity(abstracts[absind], abstracts[i], type)
        elif mattype == 'jaccard':
            simvalues[i] = 1.0 - jaccard_index(abstracts[absind], abstracts[i], type)
    return simvalues            

def master(comm, absind, abstracts, type, mattype):
    '''
    Master function for the MPI implementation to find similarity for the 
    cosine distance and jaccard distance between a given abstract (given by 
    id, absind) and all abstracts based on their "type" values
    '''
    # initialize variables
    size = comm.Get_size()
    status = MPI.Status()
    simvalues = np.float64(np.zeros(len(abstracts)))

    # Send pair of abstracts for similarity calculation
    ind = 0
    for i in range(len(abstracts)):
        if ind < size-1:
            comm.send((abstracts[absind], abstracts[i], type, mattype), dest=ind+1, tag=ind)
            ind += 1
        else:
            simval = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
            simvalues[status.Get_tag()] = 1.0 - simval
            comm.send((abstracts[absind], abstracts[i], type, mattype), dest=status.Get_source(), tag=ind)
            ind += 1
                
    # tell slaves when there are no abstracts left
    for rank in range(1,size):
        simval = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
        simvalues[status.Get_tag()] = 1.0 - simval
        comm.send((None, None, None, None), dest=status.Get_source(), tag=ind)
    
    #print "similar", mattype, simvalues
    return simvalues

def slave(comm):
    '''
    Slave function for the MPI implementation to find similarity for the 
    cosine and jaccard distance between two abstracts (abs1 and abs2) based 
    on "type" values
    '''
    status = MPI.Status()

    while True:
        # get message
        abs1, abs2, type, mattype = comm.recv(source=0, tag = MPI.ANY_TAG, status=status)
        
        # end if done
        if not abs1:
            break

        simval = 0.0        
        if mattype == "cossim": 
            simval = cosine_similarity(abs1, abs2, type)
        elif mattype == "jaccard":
            simval = jaccard_index(abs1, abs2, type)
        
        # send abstract back to master
        comm.send(simval, dest=0, tag=status.Get_tag())
