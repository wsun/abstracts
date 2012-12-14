import os
import sys
from sys import argv
import numpy as np
from abstract import Abstract
import process as Process
from mpi4py import MPI
try:
    import cPickle as pickle
except:
    import pickle

def printall(abstract):
    '''Print basic information about an article'''
    print "\nTitle: "
    print abstract.Get("title")
    print "\nAbstract: "
    print abstract.Get("text")
    print "\nTags: "
    print abstract.Get("tags")

if __name__ == '__main__': 
    # MPI values
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # check input
    version = 'p'
    type = 'bow'
    mattype = 'cossim'
    if len(sys.argv) not in range(2,6):
        if rank == 0:
            print 'Usage: ' + sys.argv[0] + ' filename' + ' [version]' + ' [frequency matrix type]' + ' [similarity measure]'
            sys.exit(0)
        else:
            sys.exit(0)

    filename = sys.argv[1]   
    if len(sys.argv) >= 3:
        version = sys.argv[2]
    if len(sys.argv) >= 4:
        type = sys.argv[3]
    if len(sys.argv) == 5:
        mattype = sys.argv[4]

    # Load all abstracts
    abstracts = []
    if os.path.isfile(filename[:-4]+"processed"):
        if rank == 0:
            pickledabs = open(filename[:-4]+"processed", "rb")
            abstracts = pickle.load(pickledabs)
            pickledabs.close()
    else:
        if version.lower() == 'p':
            abstracts = Process.main_parallel(comm, filename)
        else:
            if rank == 0:
                abstracts = Process.main_serial(comm, filename)
    
    # List 5 random abstracts and see if any are interesting
    ind = 0
    if rank == 0:
        ans = 5
        while int(ans) == 5:
            randabs = [np.random.random_integers(0, len(abstracts)-1) for i in range(5)]
            for ind, rand in enumerate(randabs):
                print str(ind) + ": " + str(abstracts[rand].Get("title")) + "\n"
            print str(len(randabs)) + ": See more ..."
            print "Are any of these articles interesting? (Enter number)\n"
            ans = raw_input()
            while not ans.isdigit():
                print "Please enter again:"
                ans = raw_input()
            while int(ans) not in range(0,6):
                print "Please enter again:"
                ans = raw_input()
            if int(ans) != 5:
                ind = randabs[int(ans)]
        
        abstract = abstracts[ind]
        printall(abstract)

    while True:
        # Ask if user wants to see similar abstracts
        if rank == 0:
            print "\n\nSee similar abstracts? (Y/N)"
            answer = raw_input().lower()
            while answer != "y" and answer != "n":
                print "Not Y or N! See similar abstracts? (Y/N)"
                answer = raw_input().lower()
            # if not, exit
            if answer == "n":
                if version.lower() == 'p':
                    for i in range(1,size):
                        comm.send(0, dest = i)
                sys.exit("Thanks for visiting!")
        # Calculate similarity values for given article
        sim_matrix = []
        if version.lower() == 'p':
            if rank == 0:
                for i in range(1,size):
                    comm.send(1, dest = i)
                sim_matrix = sorted(enumerate(Process.main_parallel_sim(comm, ind, abstracts, type, mattype)), key=lambda ind:ind[1])
            else:
                tosend = comm.recv(source = 0)
                if tosend == 1:
                    Process.main_parallel_sim(comm, ind, abstracts, type, mattype)
                elif tosend == 0:
                    sys.exit()
        else:
            if rank == 0:
                sim_matrix = sorted(enumerate(Process.main_serial_sim(comm, ind, abstracts, type, mattype)), key=lambda ind:ind[1])
            else:
                sys.exit()
        # print 5 most similar articles
        if rank == 0:
            print "Similar articles:\n"
            setabs = 0
            while True:
                if len(sim_matrix)/5 <= setabs+1:
                    print "No more articles to see ..."
                    if version.lower() == 'p':
                        for i in range(1,size):
                            comm.send(0, dest = i)
                    sys.exit("Thanks for visiting!")
                for i in range(5):
                    ind, val = sim_matrix[i+setabs*5+1]
                    print str(i) + ": " + str(abstracts[ind].Get("title")) + "\n"
                print "5: See more ...\n"
                print "6: Exit\n"
                print "What do you want to do? (Enter number)\n"
                answer = raw_input()
                while not answer.isdigit():
                    print "Please enter again:"
                    answer = raw_input()
                    print ans
                while int(answer) not in range(0,7):
                    print "Please enter again:"
                    answer = raw_input()
                if answer != "5":
                    break
                else:
                    print "More articles:\n"
                    setabs += 1
            if answer == "6":
                if version.lower() == 'p':
                    for i in range(1,size):
                        comm.send(0, dest = i)
                sys.exit("Thanks for visiting!")
            else:
                ind, val = sim_matrix[int(answer)+1]
                printall(abstracts[ind])

