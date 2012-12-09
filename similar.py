# CS205 Final Project
# Janet Song and Will Sun
#
# Calculate similarity between documents.
# 

import abstract as Abstract
import numpy as np

# Find cosine similarity for two given abstracts
def cosine_similarity(abs1, abs2, type):
    text1 = abs1.Get(type)
    text2 = abs2.Get(type)
    num = 0.0
    for ind, count in text1:
        if ind in text2:
            num += count*text2[ind]
    denom = np.linalg.norm(text1.values())*np.linalg.norm(text2.values())
    return num/denom

# Find jaccard index for two given abstracts
def jaccard_index(abs1, abs2, type):
    text1 = abs1.Get(type)
    text2 = abs2.Get(type)
    sameattr = 0
    numofattr = 0
    for ind, count in text1:
        if ind in text2:
            sameattr += 1
        numofattr += 1
    return sameattr/numofattr
    
# Find similarity matrices
def calculate_similarity_matrices(abstracts, type):
    cossim_matrix = np.zeros((len(abstracts), len(abstracts))
    jaccard_matrix = cossim_matrix
    for i in range(len(abstracts)):
        cossim_matrix[i,i] = jaccard_matrix[i,i] = 1
        for j in range(i+1,len(abstracts)):
            cossim_matrix[i,j] = cossim_matrix[j,i] = cosine_similarity(abstracts[i], abstracts[j], type)
            jaccard_matrix[i,j] = jaccard_matrix[j,i] = jaccard_Index(abstracts[i], abstracts[j], type)
    return cossim_matrix, jaccard_matrix            

# Master
def master(comm, abstracts, type):
    # initialize variables
    size = comm.Get_size()
    status = MPI.Status()
    cossim_matrix = np.zeros((len(abstracts), len(abstracts))
    jaccard_matrix = cossim_matrix
    
    # Send pair of abstracts for similarity calculation
    ind = 0
    for i in range(len(abstracts)):
        for j in range(i+1, len(abstracts)):
            if ind < size-1:
                comm.send((abstracts[i], abstracts[j], type, i,j), dest=ind+1)
                ind += 1
            else:
                cossim, jaccard, x, y = comm.recv(source=MPI.ANY_SOURCE, status=status)
                cossim_matrix[x,y] = cossim_matrix[y,x] = cossim
                jaccard_matrix[x,y] = jaccard_matrix[y,x] = jaccard
                comm.send((abstracts[i], abstracts[j], type, i, j), dest=status.Get_source())
                
    # tell slaves when there are no abstracts left
    for rank in range(1,size):
        cossim, jaccard, x, y = comm.recv(source=MPI.ANY_SOURCE, status=status)
        cossim_matrix[x,y] = cossim_matrix[y,x] = cossim
        jaccard_matrix[x,y] = jaccard_matrix[y,x] = jaccard
        comm.send((None, None, None, None, None), dest=status.Get_source())
    
    return cossim_matrix, jaccard_matrix


# Slave
def slave(comm):
    status = MPI.Status()

    while True:
        # get message
        abs1, abs2, type, i, j = comm.recv(source=0, status=status)
        
        # end if done
        if not row:
            break
         
         cossim = cosine_similarity(abs1, abs2, type)
         jaccard = jaccard_index(abs1, abs2, type)
        
        # send abstract back to master
        comm.send((cossim, jaccard, i, j), dest=0)
