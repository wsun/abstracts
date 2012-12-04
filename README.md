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

### Setup (on Mac)
- You may be able to just complete the virtualenv and pip installation, unzip the final folder, and then run the below command (and test with `yolk`); make sure you're in the git repo directory

    source final/bin/activate
    yolk -l
- Use pip and virtualenv

    sudo easy_install pip
    sudo pip install virtualenv
    virtualenv ENVIRONMENT_NAME
    source ENVIRONMENT_NAME/bin/activate

- To deactivate a virtualenv:

    deactivate

- Install Numpy and Scipy: [link](http://www.scipy.org/Installing_SciPy/Mac_OS_X); you may need to install FORTRAN compilers as noted in the link

    pip install numpy
    pip install scipy

- Install Yolk to see what packages you have

    pip install yolk
    yolk -l

- Install Beautiful Soup for scraping

    pip install beautifulsoup4

- Install lxml for faster web-crawling

    pip install lxml