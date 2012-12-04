# CS205 Final Project

### Structure
- Scraper [MPI] --> WILL
- Document class
- Process [MRJob]
    * Document similarity filtering
    * Bag of words, bigrams
    * Remove stopword / bigrams with both stopwords
    * TF-IDF
- Similarity [MRJob]
    * Cosine similarity; Jaccard index
- Topic Modeling [MPI and Gensim]
    * LSA; LDA

### Links
- Clustering to evaluate? Measure with purity and entropy
    * [one](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5571521)
    * [two](http://favi.com.vn/wp-content/uploads/2012/05/pg049_Similarity_Measures_for_Text_Document_Clustering.pdf)

- [tips](http://stackoverflow.com/questions/2380394/simple-implementation-of-n-gram-tf-idf-and-cosine-similarity-in-python?rq=1)