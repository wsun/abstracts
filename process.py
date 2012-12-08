# CS205 Final Project
# Janet Song and Will Sun
#
# Process abstracts for similarity analysis.
# 
# Bag-of-words and bigram representations, also stopword removal and tf-idf
# calculation

import re
import sys
from sys import argv
import csv
from collections import defaultdict
import numpy as np
import scipy as sp
import math

#from mrjob.job import MrJob

from abstract import Abstract
#import wf_mrjob.wordFrequency as wordFreq
#import bow_mrjob.bowFrequency as bowFreq
#import bigram_mrjob
#import stopword_mrjob
#import tfidf_wf_mrjob.tfidfwfFrequency as tfidfwfFreq

# Serial load
def load(filename, abstracts, dictionary):
    dictlist = []
    
    # load stop words
    stop_file = 'stopwords.txt'
    stops = set()

    with open(stop_file, 'rU') as stopFile:
        for row in stopFile.readlines():
            stops.add(row.replace('\n', ''))
            
    with open(filename) as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in abstracts:
                abs = Abstract(row[0])
                abs.Set('title', row[1])
                abs.Set('text', row[2][10:])
                abs.Set('tags', row[3].split(','))
                
                # remove stop words and clean text
                abstext = [''.join([c.lower() for c in word if c.isalnum()]) for word in row[2][10:].split() if word not in stops]
                abs.Set('cleantext', abstext)
                
                # add to dictionary
                for word in abstext:
                    dictlist.append(word)
                
                abstracts.append(abs)

    # remove words that only appear once
    dictlist = [word for word in dictlist if dictlist.count(word) > 1]
    for word in dictlist:
        if word not in dictionary:
            dictionary.append(word)
    dictionary.sort()

# Serial bag of words
def bagofwords(abstracts, dictionary):
    for abstract in abstracts:
        bow = defaultdict(float)
        abstext = abstract.Get('cleantext')
        for word in abstext:
            if word in dictionary:
                ind = dictionary.index(word)
                bow[ind] += 1.0
        normalize(bow)
        abstract.Set('bow', bow)

# Serial bigrams
def bigram(abstracts, dictionary):
    for abstract in abstracts:
        bigram = defaultdict(float)
        abstext = abstract.Get('cleantext')
        for i in range(len(abstext)-1):
            wordgram = abstext[i:i+2]
            wordgram.sort()
            if wordgram[0] in dictionary:
                if wordgram[1] in dictionary:
                    pair = (dictionary.index(wordgram[0]),dictionary.index(wordgram[1]))
                    bigram[pair] += 1.0
        normalize(bigram)
        abstract.Set('bigram',bigram)

# Serial TFIDF for bag of words or bigrams
def tfidf(abstracts, dictionary, type):
    termdoc = termall(abstracts, type)
    numabs = float(len(abstracts))
    for abstract in abstracts:
        tfidf = defaultdict(int)
        for ind, freq in abstract.Get(type).iteritems():
            tfidf[ind] = freq*math.log(numabs/termdoc[ind])
        abstract.Set('tfidf'+type, tfidf)

# Find number of documents in which a phrase or word appears
def termall(abstracts, type):
    termall = defaultdict(float)
    for abstract in abstracts:
        for ind, count in abstract.Get(type).iteritems():
            termall[ind] += 1.0
    return termall

# Serial normalize
def normalize(array):
    numwords = float(sum(array.values()))
    for ind, count in array.iteritems():
        array[ind] = count/numwords
    return array
    

if __name__ == '__main__':

    gramsize = 2
    abstracts = []
    dictionary = []
    wordlist = []
    
    # serial version
    script, filename = argv
    load(filename, abstracts, dictionary)   
    # create dict of word frequency (bag of words)
    bagofwords(abstracts, dictionary)
    # create dict of bigram frequency
    bigram(abstracts, dictionary)
    # create dict of tfidf
    tfidf(abstracts, dictionary, 'bow')
    tfidf(abstracts, dictionary, 'bigram')
    #for abstract in abstracts:
    #    print abstract.Get('tfidfbow')
    #    print abstract.Get('bigram')

    # parallel version
