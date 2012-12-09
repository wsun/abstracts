# CS205 Final Project
# Janet Song and Will Sun
#
# Calculate similarity between documents.
# 

import abstract as Abstract
import numpy as np
from mpi4py import MPI

# Find cosine similarity for two given abstracts
def cosine_similarity(abs1, abs2, type):
    text1 = abs1.Get(type)
    text2 = abs2.Get(type)
    num = float(0)
    for ind, count in text1.iteritems():
        if ind in text2:
            num += float(count*text2[ind])
    denom = float(np.linalg.norm(text1.values())*np.linalg.norm(text2.values()))
    return float(num/denom)

# Find jaccard index for two given abstracts
def jaccard_index(abs1, abs2, type):
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
    
# Find similarity matrices (Serial)
def calculate_similarity_matrices(absind, abstracts, type):
    cossim = np.float64(np.zeros(len(abstracts)))
    jaccard = np.float64(np.zeros(len(abstracts)))
    for i in range(len(abstracts)):
        cossim_matrix[i] = 1 - cosine_similarity(abstracts[absind], abstracts[i], type)
        jaccard_matrix[i] = 1 - jaccard_index(abstracts[absind], abstracts[i], type)
    return cossim_matrix, jaccard_matrix            

# Master, Find similarity matrices (Parallel)
def master(comm, absind, abstracts, type):
    # initialize variables
    size = comm.Get_size()
    status = MPI.Status()
    cossim = np.float64(np.zeros(len(abstracts)))
    jaccard = np.float64(np.zeros(len(abstracts)))

    # Send pair of abstracts for similarity calculation
    ind = 0
    for i in range(len(abstracts)):
        if ind < size-1:
            comm.send(abstracts[absind], abstracts[i], type), dest=ind+1, tag=ind)
            ind += 1
        else:
            cossim, jaccard = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
            cossim_matrix[status.Get_tag()] = 1 - cossim
            jaccard_matrix[status.Get_tag()] = 1 - jaccard
            comm.send((abstracts[absind], abstracts[i], type), dest=status.Get_source(), tag=ind)
            ind += 1
                
    # tell slaves when there are no abstracts left
    for rank in range(1,size):
        cossim, jaccard = comm.recv(source=MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
        cossim_matrix[status.Get_tag()] = 1 - cossim
        jaccard_matrix[status.Get_tag()] = 1 - jaccard
        comm.send((None, None, None), dest=status.Get_source(), tag=ind)
    
    #print "similar", cossim_matrix
    return cossim_matrix, jaccard_matrix


# Slave
def slave(comm):
    status = MPI.Status()

    while True:
        # get message
        abs1, abs2, type = comm.recv(source=0, tag = MPI.ANY_TAG, status=status)
        
        # end if done
        if not abs1:
            break
         
        cossim = cosine_similarity(abs1, abs2, type)
        jaccard = jaccard_index(abs1, abs2, type)
        
        # send abstract back to master
        comm.send((cossim, jaccard), dest=0, tag=status.Get_tag())
