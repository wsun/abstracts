## Setup

Setup (on Mac)

- Download files
- Install pip, virtualenv, and the list of packages, test it out with yolk.

        sudo easy_install pip
        sudo pip install virtualenv
        virtualenv final [--no-site-packages]
        source final/bin/activate
    
Setup (on VM)

- Download files
- In a `RESONANCE NODE` or `RESONANCE HEADNODE`:
    
        echo "module load packages/epd/7.1-2" >> ~/.bashrc

- reload shell
    
        virtualenv final
        source final/bin/activate

- To deactivate a virtualenv:

        deactivate

Packages (*install these packages on VM)

1. Numpy and Scipy: you may need to install FORTRAN compilers as noted in the link (http://www.scipy.org/Installing_SciPy/Mac_OS_X)

        pip install numpy
        pip install scipy

2. Yolk to see what packages you have*
        
        pip install yolk
        yolk -l

3. Beautiful Soup for scraping*
        
        pip install beautifulsoup4

4. lxml for faster html parsing

        pip install lxml

5. Mechanize for web-navigation*
        
        pip install mechanize

6. Gensim for topic modeling*
        
        pip install gensim

7. nltk for K-means clustering*
        
        pip install nltk

## How to run
To run the scraper:
    
    mpirun -n P python scraper.py [topic] [startpage] [-m, -s, -x]
    OR
    qsub runscript

    - P: number of processes
    - topic: subject query to be entered into Web of Science
    - startpage: first page of results to explore (e.g. if 1000, then scraper will explore pages 1001-2000, with default setting for n = 1000 pages)
    - (-m) test masterslave
    - (-s) test scattergather
    - (-x) just scrape, no timings or checks
    Ex. mpirun -n 4 python scraper.py biology 1000 -x

To run the shell:

    mpirun -n P python findabstracts.py data [implementation] [type] [similarity type]    
    - P: number of processes
    - data: data location (e.g. data/clustertest.csv)
    - implementation: 'p' for the master-slave MPI implementation, 
                      'g' for the scatter-gather MPI implementation, or
                      's' for the serial implementation
    - type: 'bow' for bag of words frequency matrix,
            'bigram' for bigram frequency matrix,
            'tfidfbow' for weighted bag of words frequency matrix,
            'tfidfbigram' for weighted bigram frequency matrix,
            'lda' for Latent Dirichlet Allocation (LDA) topic modeling, or
            'lsa' for Latent Semantic Analysis (LSA) topic modeling
    - similarity type: 'cossim' for cosine distance or
                       'jaccard' for jaccard distance
    - implementation, type, and similarity type are optional
    - default values are master-slave MPI  implementation, bag of words frequency matrix, and cosine distance measure
    
    Ex. mpirun -n 4 python findabstracts.py data/bio0-2000.csv g bow cossim

NOTE: If the data has already been processed before, the shell will simply unpickle the processed data instead of processing all of the abstracts.  For convenience, a pickled version of 2000 biology Abstracts is provided in data/bio0-2000processed.

To run the evaluation:

    python cluster.py [filename] [type] [similarity type]
    - filename: filename of csv file to use for abstracts
    - type: 'bow', 'bigram', 'lsa', 'lda'; evaluate this representation
    - similarity type: 'euc' for euclidean distance, 'cos' for cosine distance,
            'jac' for jaccard distance
    

## Code
`abstract.py`

- contains Abstract class that is used to wrap information on each abstract

`cluster.py`

- contains evaluation code to explore different data representations and similarity metrics
- uses K-means clustering and various evaluation strategies, including sum of distances, purity, entropy, Rand index, and F1 measure

`findabstracts.py`

- shell to allow for user interaction with abstracts
- presents users with articles and can display information on each article and 
  show similar articles

`ldamodel.py`

- main LDA model code, modified from gensim's implementation to involve MPI

`lda.py`

- serial and parallel implementation, with MPI master/slave, of LDA model generation
- can be run individually on saved dictionary ('xxx.dict') and corpus ('xxx.mm', 'xxx.mm.index') for timing comparison

`lsa.py`

- serial and parallel implementation, with MPI master/slave, of LSA model generation
- can be run individually on saved dictionary ('xxx.dict') and corpus ('xxx.mm', 'xxx.mm.index') for timing comparison

`lsimodel.py`

- main LSA model code, modified from gensim's implementation to involve MPI

`process.py`

- Parallel (MPI master-slave and scatter-gather) and serial implementations
- loads the data into Abstract classes
- Finds unweighted and weighted (TF-IDF) bag of words and bigram frequency matrices for each abstract
- Uses topic modeling with LDA and LSA to generate topic matrices for each abstract
- Calculates cosine or jaccard distance between a given abstract and all other abstracts using similar.py

`processtest.py`

- times parallel and serial versions of process.py 

`scraper.py`

- web scraper that searches and scrapes Web of Science database
- uses Mechanize, lxml, and BeautifulSoup to make queries and parse HTML
- all queries are placed in a 'Topic' search field; results are written out to csv

`similar.py`

- Parallel (MPI master-slave) and serial implementations
- Calculates cosine or jaccard distance between a given abstract and all other abstracts


## Data
In the data folder, there are files of data scraped from Web of Science using scraper.py. Each file contains 8000 separate articles. We have also included a smaller dataset, clustertest.csv (100 articles), to allow for ease of testing, as well as a preprocessed dataset, bio0-2000processed (2000 articles).