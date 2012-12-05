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
- Make sure you're in the git repo directory; install `pip`, `virtualenv`, and the list of packages, test it out with `yolk`.

        sudo easy_install pip
        sudo pip install virtualenv
        unzip final.zip
        source final/bin/activate
        yolk -l

===============================================================================

- Making a new virtualenv:

        virtualenv ENVIRONMENT_NAME
        source ENVIRONMENT_NAME/bin/activate

- To deactivate a virtualenv:

        deactivate

- Numpy and Scipy: [link](http://www.scipy.org/Installing_SciPy/Mac_OS_X); you may need to install FORTRAN compilers as noted in the link

        pip install numpy
        pip install scipy

- Yolk to see what packages you have

        pip install yolk
        yolk -l

- Beautiful Soup for scraping

        pip install beautifulsoup4

- lxml for faster html parsing

        pip install lxml

- Mechanize for web-navigation

        pip install mechanize