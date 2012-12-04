# CS205 Final Project
# Janet Song and Will Sun
#
# Process abstracts for similarity analysis.
# 
# Bag-of-words and bigram representations, also stopword removal and tf-idf
# calculation

from abstract import Abstract
import bow
import bigram
import stopword
import tfidf

if __name__ == '__main__':

    # load in objects
    # check duplicates on a dictionary, delete files?
    # use MRJob runner to switch between bag of words and bigrams to process
    # use MRJob runner for stopword removal
    # use MRJob runner for TF-IDF