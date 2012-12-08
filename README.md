# CS205 Final Project

### Structure
- Scraper [MPI] --> eval masterslave, scattergather; descriptions of website
- Abstract class
- Process [MPI]
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

- [tips on process](http://stackoverflow.com/questions/2380394/simple-implementation-of-n-gram-tf-idf-and-cosine-similarity-in-python?rq=1)

### Setup (on Mac)
- Make sure you're in the git repo directory; install `pip`, `virtualenv`, and the list of packages, test it out with `yolk`.

        sudo easy_install pip
        sudo pip install virtualenv
        virtualenv final [--no-site-packages]
        source final/bin/activate

### Setup (on VM)
- On LOCAL MACHINE, clone the repo
- In a RESONANCE NODE or RESONANCE HEADNODE:

        echo "module load packages/epd/7.1-2" >> ~/.bashrc
        [reload shell?]
        virtualenv final
        source final/bin/activate


### Packages (*install these packages on VM)
- To deactivate a virtualenv:

        deactivate

- Numpy and Scipy: [link](http://www.scipy.org/Installing_SciPy/Mac_OS_X); you may need to install FORTRAN compilers as noted in the link

        pip install numpy
        pip install scipy

- Yolk to see what packages you have*

        pip install yolk
        yolk -l

- Beautiful Soup for scraping*

        pip install beautifulsoup4

- lxml for faster html parsing

        pip install lxml

- Mechanize for web-navigation*

        pip install mechanize

- Gensim for topic modeling*

        pip install gensim