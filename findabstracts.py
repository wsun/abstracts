import os
import sys
from sys import argv
import numpy as np
from abstract import Abstract
import process as Process
from mpi4py import MPI

def printall(abstract):
    print "\nTitle: "
    print abstract.Get("title")
    print "\nAbstract: "
    print abstract.Get("text")
    print "\nTags: "
    print abstract.Get("tags")

if __name__ == '__main__':
    script, filename, version, type, mattype = argv
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    abstracts = []
    if version.lower() == 'p':
        abstracts = Process.main_parallel(comm, filename)
    else:
        if rank == 0:
            abstracts = Process.main_serial(comm, filename)
    
    ind = 0
    if rank == 0:
        ans = 5
        while int(ans) == 5:
            randabs = [np.random.random_integers(0, len(abstracts)-1) for i in range(5)]
            for ind, rand in enumerate(randabs):
                print str(ind) + ": " + str(abstracts[rand].Get("title")) + "\n"
            print str(len(randabs)) + ": See more ..."
            print "Are any of these articles interesting?\n"
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
        if rank == 0:
            print "\n\nSee similar abstracts? (Y/N)"
            answer = raw_input().lower()
            while answer != "y" and answer != "n":
                print "Not Y or N! See similar abstracts? (Y/N)"
                answer = raw_input().lower()
            if answer == "n":
                if version.lower() == 'p':
                    for i in range(1,size):
                        comm.send(0, dest = i)
                sys.exit("Thanks for visiting!")
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
                print "What do you want to do?\n"
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
                ind, val = sim_matrix[int(answer)]
                printall(abstracts[ind])

