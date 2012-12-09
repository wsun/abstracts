import os
import sys
from sys import argv
import numpy as np
from abstract import Abstract
import process_find as Process

if __name__ == '__main__':
    script, filename = argv
    abstracts, sim_matrix = Process.main(filename)

    ans = 5
    ind = 0
    while ans == 5:
        randabs = [np.random.random_integers(0, len(abstracts)-1) for i in range(5)]
        print "Are any of these articles interesting?\n",
        for ind, rand in enumerate(randabs):
            print ind + ": " + abstracts[rand] + "\n"
        print len(randabs) + ": See more ..."
        ans = raw_input()
        if ans != 5:
            ind = randabs[ans]

    abstract = abstracts[ind]
    print abstract.Get("title")
    print "\n\nAbstract: "
    print abstract.Get("text")
    print "\n\nTags: "
    print abstract.Get("tags")
    print "\n\nSee similar abstracts? (Y/N)"
    answer = raw_input().lower()
    while answer != "y" or answer != "n":
        print "Not Y or N! See similar abstracts? (Y/N)"
        answer = raw_input().lower()
    if answer == "n":
        return
    else:
        row = sim_matrix[ind,:]
        simarts = np.zeros(5)
        for val in row:
            if row[0]
        print "Similar articles:\n"
        for i in range(5):
            print 
        
    




   
    


