import csv
import re
import sys
from gensim import corpora, models, similarities


import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s: %(message)s", level=logging.INFO)


def load(ids, docs):
    data_file = 'data.csv'
    replacements = {'1920s': 'twenties', '20s': 'twenties', 
                    '1930s': 'thirties', '30s': 'thirties',
                    '1940s': 'forties', '40s': 'forties', 
                    '1950s': 'fifties', '50s': 'fifties', 
                    '1960s': 'sixties', '60s': 'sixties',
                    '1970s': 'seventies', '70s': 'seventies', 
                    '1980s': 'eighties', '80s': 'eighties', 
                    '1990s': 'nineties', '90s': 'nineties', 
                    '2000s': 'noughties', '00s': 'noughties', 
                    '3/4': 'three quarters', 't-shirt': 'tshirt',
                    'grey': 'gray', 'colour': 'color', 'coloured': 'colored',
                    'cami': 'camisole', 'oversized': 'oversize',
                    'boho': 'bohemian', 'strapped': 'strappy'}
    x = []

    # read the csv
    with open(data_file, 'rU') as csvfile:
        outfitreader = csv.reader(csvfile)
        for row in outfitreader:
            ids.append(int(row[0]))
            clean = ' '.join([row[1], row[2]]).strip()
            
            # replace times
            for key, value in replacements.iteritems():
                clean = clean.replace(key, value)

            # tokenize and store
            clean = re.sub('[^a-zA-Z0-9]', ' ', clean)
            x.append(clean.lower().split())

    # read the default
    y = []
    for i in ids:
        default_file = '../img/' + str(i) + '/' + str(i) + '.txt'
        document = []
        tmp = []

        with open(default_file, 'rU') as textFile:
            for row in textFile.readlines():
                document.append(row)

        # fill in text
        tmp.append(document[2]) # title
        count = 0
        if (document[8] != '----------\n'): # tags
            while (document[8 + count] != '----------\n'):
                tmp.append(document[8 + count])
                count += 1
        if (document[8 + 2 + count] != '----------\n'): # colors
            while (document[8 + 2 + count] != '----------\n'):
                tmp.append(document[8 + 2 + count])
                count += 1
        if (document[8 + 4 + count] != '----------\n'): # apparel
            while (document[8 + 4 + count] != '----------\n'):
                tmp.append(document[8 + 4 + count])
                count += 1
        if (document[8 + 6 + count] != '----------\n'): # comments
            while (document[8 + 6 + count] != '----------'):
                tmp.append(document[8 + 6 + count])
                if (8 + 6 + count == len(document) - 1):
                    break
                count += 1

        # clean text
        clean = ' '.join(tmp).strip()
        for key, value in replacements.iteritems():
            clean = clean.replace(key, value)
        clean = re.sub('[^a-zA-Z0-9]', ' ', clean)
        y.append(clean.lower().split())

    # combine
    for i in xrange(len(ids)):
        docs.append(x[i] + y[i])

def cleanup(ids, docs):
    stop_file = 'stopwords.txt'
    stops = set()

    with open(stop_file, 'rU') as stopFile:
        for row in stopFile.readlines():
            stops.add(row.replace('\n', ''))

    # remove stops
    docs = [[word for word in document if word not in stops] for document in docs]

    # remove words that only appear once
    all_tokens = sum(docs, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    docs = [[word for word in document if word not in tokens_once] for document in docs]
    return docs

if __name__ == '__main__':

    # parse and create corpus
    ids = []
    docs = []
    load(ids, docs)
    docs = cleanup(ids, docs)
    total = docs[:]

    # create holdout set
    holdout = []
    for i in xrange(10):
        holdout.append(docs.pop())

    # create dictionary: mapping from features to their integer ids
    dictionary = corpora.Dictionary(total)
    # dictionary.save('test.dict')
    
    # create corpus: sparse vector for each document
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    corpush = [dictionary.doc2bow(doc) for doc in holdout]
    # corpora.MmCorpus.serialize('test.mm', corpus)

    '''
    # load dictionary and corpus
    dictionary = corpora.Dictionary.load('test.dict')
    corpus = corpora.MmCorpus('test.mm')
    '''

    # prepare TFIDF
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    tfidfh = models.TfidfModel(corpush)
    corpush_tfidf = tfidfh[corpush]

    '''
    # prepare LDA
    lda = models.LdaModel(corpus_tfidf, num_topics=20, id2word=dictionary, passes=100, update_every=0)
    corpus_lda = lda[corpus_tfidf]
    # lda.show_topics()    
    '''
    
    '''
    # diagnostics
    count = 0
    for i in xrange(10):
        lda = models.LdaModel(corpus_tfidf, num_topics=16, id2word=dictionary, passes=100, update_every=0)
        count += lda.bound(corpush_tfidf)
    print count / 10
    '''

    
    # prepare LSI
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=12)
    corpus_lsi = lsi[corpus_tfidf]
    print lsi.show_topics(num_topics=12)
