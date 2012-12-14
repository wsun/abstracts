from gensim import corpora, models
from collections import defaultdict
import process as Process
import lda as Lda
import sys, random

def modified_process(filename):
    ''' Serial processing of abstracts, no topic modeling. '''
    abstracts = []
    dictionary = []

    # load stop words
    stops = set()
    stop_file = 'stopwords.txt'
    with open(stop_file, 'rU') as stopFile:
        for row in stopFile.readlines():
            stops.add(row.replace('\n', ''))
    
    dictlist = Process.load(filename, abstracts, stops) 
    # create dictionary
    Process.create_dict(dictlist, dictionary)

    # clean text of words not in dictionary
    for abstract in abstracts:
        abstext = [word for word in abstract.Get('cleantext') if word in dictionary]
        abstract.Set('cleantext', abstext)

    dictlength = len(dictionary) 
    bigramdict = []
    termbow = defaultdict(float)
    termbigram = defaultdict(float)
    for abstract in abstracts:
        # create dict of word frequency (bag of words)
        bow = Process.create_bagofwords(abstract, dictionary)
        abstract.Set('bow', bow)
        abstract.Set('bownum', dictlength)
        for ind in bow.keys():
            termbow[ind] += 1.0
        # create dict of bigram frequency
        bigram, bigramdict = Process.create_bigram(abstract, dictionary, bigramdict)
        abstract.Set('bigram', bigram)
        for pair in bigram.keys():
            termbigram[pair] += 1.0
    # create dict of tfidf
    Process.serial_tfidf(abstracts, 'bow', termbow, len(bigramdict))
    Process.serial_tfidf(abstracts, 'bigram', termbigram)

    return abstracts

def perplexity(abstracts, nums):
    ''' Serial computation of topic models for all abstracts. '''
    # prepare dictionary and corpora for topic modeling
    docs = [abstract.Get('cleantext') for abstract in abstracts]
    dictionary = corpora.Dictionary(docs)
    #dictionary.save('abstracts.dict')           

    # main loop
    random.seed()
    for num in nums:
        count = 0

        for i in xrange(3):
            # prepare holdout set
            p = range(len(docs))
            random.shuffle(p)
            docs = [docs[i] for i in p]
            tenth = int(len(docs) / 10)
            train = docs[tenth:]
            test = docs[:tenth]

            traincorpus = [dictionary.doc2bow(doc) for doc in train]
            testcorpus = [dictionary.doc2bow(doc) for doc in test]

            traintfidf = models.TfidfModel(traincorpus)
            testtfidf = models.TfidfModel(testcorpus)

            traincorpus2 = traintfidf[traincorpus]
            testcorpus2 = testtfidf[testcorpus]

            ldaModel = Lda.serial(traincorpus2, dictionary, num, chunksize=1000, alpha=50.0/num, eta=2.0/num)
            count += ldaModel.bound(testcorpus2)

        avg = count / 3.0
        print "%d: %f" % (num, avg)

    return

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'Usage: ' + sys.argv[0] + ' filename'
        sys.exit(0)

    filename = sys.argv[1]
    abstracts = modified_process(filename)
    nums = [4,6,8,10,12,14,16,18,20,30,40,50]
    perplexity(abstracts, nums)
