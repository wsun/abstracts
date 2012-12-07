# CS205 Final Project
# Janet Song and Will Sun
#
# Process abstracts for similarity analysis.
# 
# Bag-of-words and bigram representations, also stopword removal and tf-idf
# calculation

from sys import argv
import csv
from collections import defaultdict
import numpy as np
import scipy as sp
import math

from mrjob.job import MrJob

from abstract import Abstract
#import wf_mrjob.wordFrequency as wordFreq
#import bow_mrjob.bowFrequency as bowFreq
#import bigram_mrjob
#import stopword_mrjob
#import tfidf_wf_mrjob.tfidfwfFrequency as tfidfwfFreq

if __name__ == '__main__':

    abstracts = defaultdict(Abstract)
    stopwords = ['the', 'a', 'an', 'is']
    gramsize = 2
    termdoc = defaultdict(int)
    phrasedoc = defaultdict(int)
    
    # load in objects
    
    # serial version
    script, filename = argv
    with open(filename) as csvfile:
        scrapedata = csv.reader(csvfile)
        for row in scrapedata:
            # check if duplicate
            if row[0] not in abstracts:
                abs = Abstract(row[0])
                abs.Set('title', row[1])
                abs.Set('text', row[2][10:])
                abs.Set('tags', row[3])
                
                abstext = abs.Get('text').split()
                
                # remove stopwords
                for stopword in stopwords:
                    while stopword in abstext:
                        abstext.remove()
                abs.Set('numwords', len(abstext))
                
                # create dict of word frequency (bag of words)
                bow = defaultdict(float)
                for word in abstext:
                    word = word.join([c for c in word if c.isalnum()])
                    if word not in bow:
                        bow[word].append(1.0)
                        # overall word frequency in documents
                        if word not in termdoc:
                            termdoc[word].append(1)
                        else:
                            termdoc[word] += 1
                    else:
                        bow[word] += 1.0
                # normalize
                for word, count in bow:
                    bow[word] = count/sabs.Get('numwords')
                abs.Set('bow', bow)
                
                # create dict of n-grams
                bigram = defaultdict(float)
                for i in range(len(abstext)-gramsize+1):
                    wordgram = abstext[i:i+gramsize].sort()
                    if wordgram not in bigram:
                        bigram[wordgram].append(1.0)
                        # overall phrase frequency in documents
                        if wordgram not in termdoc:
                            phrasedoc[wordgram].append(1)
                        else:
                            phrasedoc[wordgram] += 1
                    else:
                        bigram[wordgram] += 1.0
                # normalize
                for phrase, count in bigram:
                    bigram[phrase] = count/sp.misc.comb(abs.Get('numwords'), gramsize)
                abs.Set('bigram',bigram)
               
                abstracts[abs.Get('path')].append(abs)
'''      
    for abstract in abstracts:
        # create dict of word frequency using TF-IDF
        normwf = defaultdict(int)
        for word, freq in abs.Get('wf'):
            normwf[word].append(freq*math.log(float(termdoc[word]/len(termdoc))))
        abs.Set('tfidfwf', normwf)
    
        # create dict of bags of words using TF-IDF 
        normbow = defaultdict(int)
        for bag, freq in abs.Get('bow'):
            normwf[bag].append(freq*math.log(float(phrasedoc[bag]/len(phrasedoc))))
        abs.Set('tfidfbow', normbow)
'''    
    # parallel version
    
    
    # use MRJob runner to switch between bag of words and bigrams to process
    # use MRJob runner for stopword removal
    # use MRJob runner for TF-IDF