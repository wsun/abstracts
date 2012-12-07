# CS205 Final Project
# Janet Song and Will Sun
#
# Scraper for Web of Science using MPI
# 
# Use provided runscript

from mpi4py import MPI
import sys, csv, os, time, random, math, itertools
import urllib2
import mechanize
from bs4 import BeautifulSoup

# User-Agents
agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.95 Safari/537.11',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/536.26.17 (KHTML, like Gecko) Version/6.0.2 Safari/536.26.17',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.4 (KHTML, like Gecko) Chrome/22.0.1229.94 Safari/537.4',
         ]

randInt = 5     # maximum sleep
PAGES = 10      # maximum pages per query

def slave(comm, topic):
    status = MPI.Status()

    # setup the browser
    br = mechanize.Browser()
    br.set_handle_equiv(True)       # handle HTTP-EQUIV headers
    br.set_handle_redirect(True)    # handle redirects
    br.set_handle_referer(True)     # referer header
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
                                    # follow refresh 0, no hangs on refresh > 0
    br.set_handle_robots(False)     # screw the robots.txt
    agent = random.choice(agents)
    br.addheaders = [('User-Agent', agent)]

    # debug
    '''
    br.set_debug_http(True)
    br.set_debug_redirects(True)
    br.set_debug_responses(True)
    '''

    # make the request, go to Web of Science
    br.open('http://apps.webofknowledge.com')
    time.sleep(random.random())
    br.follow_link(br.find_link(url_regex='WOS_GeneralSearch_input'))
    br.select_form(nr=0)
    br['value(input1)'] = topic
    br.submit()

    # find the key URL
    urlArray = list(br.find_link(url_regex='page=2').url)
    
    # MAIN WORK
    while True:
        page = comm.recv(source=0)
        if (page == None):
            break
        else:
            # begin processing the page
            urlArray[-1] = str(page)
            url = ''.join(urlArray)
            br.open(url)
            soup = BeautifulSoup(br.response().read(), 'lxml')

            links = []
            dois = []
            titles = []
            abstracts = []
            categories = []

            # build the ten links
            for i in xrange(10):
                articleId = ((page - 1) * 10) + i + 1
                x = soup.select("#RECORD_" + str(articleId))
                article = x[0]

                # capture the link
                links.append('http://apps.webofknowledge.com' + article.select(".smallV110")[0]["href"])

            # navigate to get abstracts and categories
            for l in links:
                time.sleep(random.random())

                r2 = urllib2.Request(l)
                r2.add_header('User-Agent', agent)
                response2 = urllib2.urlopen(r2)
                soup2 = BeautifulSoup(response2, 'lxml')

                if soup2.find(text="Abstract:") and soup2.find(text="DOI:") and soup2.select(".FullRecTitle") and soup2.find(text="Web of Science Categories:"):
                    dois.append(soup2.find(text="DOI:").next_element.next_element.next_element.encode('utf-8'))
                    titles.append(soup2.select(".FullRecTitle")[0].value.text.encode('utf-8'))
                    abstracts.append(soup2.find(text="Abstract:").parent.parent.text.encode('utf-8'))
                    categories.append(soup2.find(text="Web of Science Categories:").next_element.encode('utf-8'))

                time.sleep(random.randint(1, randInt))

            total = [dois, titles, abstracts, categories]
            comm.send(total, dest=0, tag=page)

    return


def master(comm, topic):
    size = comm.Get_size()
    status = MPI.status()
    results = {}

    count = 0
    for i in xrange(PAGES):
        real = i + 1
        # initial distribution to slaves
        if (i < size - 1):
            comm.send(real, dest=i+1)

        # wait for completion
        else:
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            count += 1
            results[tag] = result
            comm.send(real, dest=source)

    # wait for stragglers
    while (count < PAGES):
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        count += 1
        results[tag] = result

    # kill the slaves
    for p in xrange(size):
        if p != 0:
            comm.send(None, dest=p)

    # collect the results
    dois = []
    titles = []
    abstracts = []
    categories = []
    total = []
    for i, v in results.iteritems():
        dois.extend(v[0])
        titles.extend(v[1])
        abstracts.extend(v[2])
        categories.extend(v[3])
    for i in xrange(len(dois)):
        tmp = [dois[i], titles[i], abstracts[i], categories[i]]
        total.append(tmp)

    return total

def serial(topic):
    # setup the browser
    br = mechanize.Browser()
    br.set_handle_equiv(True)       # handle HTTP-EQUIV headers
    br.set_handle_redirect(True)    # handle redirects
    br.set_handle_referer(True)     # referer header
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
                                    # follow refresh 0, no hangs on refresh > 0
    br.set_handle_robots(False)     # screw the robots.txt
    agent = random.choice(agents)
    br.addheaders = [('User-Agent', agent)]

    # debug
    '''
    br.set_debug_http(True)
    br.set_debug_redirects(True)
    br.set_debug_responses(True)
    '''

    # make the request, go to Web of Science
    br.open('http://apps.webofknowledge.com')
    time.sleep(random.random())
    br.follow_link(br.find_link(url_regex='WOS_GeneralSearch_input'))
    br.select_form(nr=0)
    br['value(input1)'] = topic
    br.submit()

    # find the key URL
    urlArray = list(br.find_link(url_regex='page=2').url)
    dois = []
    titles = []
    abstracts = []
    categories = []
    total = []
    
    # MAIN WORK
    for i in xrange(PAGES):
        page = i + 1

        # begin processing the page
        urlArray[-1] = str(page)
        url = ''.join(urlArray)
        br.open(url)
        soup = BeautifulSoup(br.response().read(), 'lxml')

        links = []

        # build the ten links
        for i in xrange(10):
            articleId = ((page - 1) * 10) + i + 1
            x = soup.select("#RECORD_" + str(articleId))
            article = x[0]

            # capture the link
            links.append('http://apps.webofknowledge.com' + article.select(".smallV110")[0]["href"])

        # navigate to get abstracts and categories
        for l in links:
            time.sleep(random.random())

            r2 = urllib2.Request(l)
            r2.add_header('User-Agent', agent)
            response2 = urllib2.urlopen(r2)
            soup2 = BeautifulSoup(response2, 'lxml')

            if soup2.find(text="Abstract:") and soup2.find(text="DOI:") and soup2.select(".FullRecTitle") and soup2.find(text="Web of Science Categories:"):
                dois.append(soup2.find(text="DOI:").next_element.next_element.next_element.encode('utf-8'))
                titles.append(soup2.select(".FullRecTitle")[0].value.text.encode('utf-8'))
                abstracts.append(soup2.find(text="Abstract:").parent.parent.text.encode('utf-8'))
                categories.append(soup2.find(text="Web of Science Categories:").next_element.encode('utf-8'))

            time.sleep(random.randint(1, randInt))

    # process the results
    for i in xrange(len(dois)):
        tmp = [dois[i], titles[i], abstracts[i], categories[i]]
        total.append(tmp)

    return total


def save(results, topic):
    f = 'data/' + topic + '.csv'
    directory = os.path.dirname(f)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


def masterslave(topic):
    # get MPI data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        p_start = MPI.Wtime()
        p_result = master(comm, topic)
        p_stop = MPI.Wtime()

        # serial implementation
        s_start = time.time()
        s_result = serial(topic)
        s_stop = time.time()

        print "Serial Time: %f secs" % (s_stop - s_start)
        print "Parallel Time: %f secs" % (p_stop - p_start)

        # check
        errors = 0
        for i in xrange(len(p_result)):
            if p_result[i][0] != s_result[i][0]: 
                print "ERROR: %d (%s, %s)" % (i, p_result[i][0], s_result[i][0])
            elif p_result[i][1] != s_result[i][1]: 
                print "ERROR: %d (%s, %s)" % (i, p_result[i][1], s_result[i][1])
            elif p_result[i][2] != s_result[i][2]: 
                print "ERROR: %d (%s, %s)" % (i, p_result[i][2], s_result[i][2])
            elif p_result[i][3] != s_result[i][3]:
                print "ERROR: %d (%s, %s)" % (i, p_result[i][3], s_result[i][3])
        print "TOTAL ERRORS: %d" % errors

        # save
        save(p_result, topic)

    else:
        slave(comm, topic)

def scattergather(topic):
    # get MPI data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.barrier()
    p_start = MPI.Wtime()

    N = int(math.ceil( PAGES / float(size) ))

    # indices of pages to scrape
    indices = []
    for i in xrange(N):
        val = (rank * N) + i
        if val < PAGES:
            indices.append(val)
        
    # setup the browser
    br = mechanize.Browser()
    br.set_handle_equiv(True)       # handle HTTP-EQUIV headers
    br.set_handle_redirect(True)    # handle redirects
    br.set_handle_referer(True)     # referer header
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
                                    # follow refresh 0, no hangs on refresh > 0
    br.set_handle_robots(False)     # screw the robots.txt
    agent = random.choice(agents)
    br.addheaders = [('User-Agent', agent)]

    # debug
    '''
    br.set_debug_http(True)
    br.set_debug_redirects(True)
    br.set_debug_responses(True)
    '''

    # make the request, go to Web of Science
    br.open('http://apps.webofknowledge.com')
    time.sleep(random.random())
    br.follow_link(br.find_link(url_regex='WOS_GeneralSearch_input'))
    br.select_form(nr=0)
    br['value(input1)'] = topic
    br.submit()

    # find the key URL
    urlArray = list(br.find_link(url_regex='page=2').url)
    dois = []
    titles = []
    abstracts = []
    categories = []
    total = []
    
    # MAIN WORK
    for i in indices:
        page = i + 1

        print "PAGE: %d" % page

        # begin processing the page
        urlArray[-1] = str(page)
        url = ''.join(urlArray)
        br.open(url)
        soup = BeautifulSoup(br.response().read(), 'lxml')

        links = []

        # build the ten links
        for i in xrange(10):
            articleId = ((page - 1) * 10) + i + 1
            x = soup.select("#RECORD_" + str(articleId))
            article = x[0]

            # capture the link
            links.append('http://apps.webofknowledge.com' + article.select(".smallV110")[0]["href"])

        # navigate to get abstracts and categories
        for l in links:
            time.sleep(random.random())

            r2 = urllib2.Request(l)
            r2.add_header('User-Agent', agent)
            response2 = urllib2.urlopen(r2)
            soup2 = BeautifulSoup(response2, 'lxml')

            if soup2.find(text="Abstract:") and soup2.find(text="DOI:") and soup2.select(".FullRecTitle") and soup2.find(text="Web of Science Categories:"):
                dois.append(soup2.find(text="DOI:").next_element.next_element.next_element.encode('utf-8'))
                titles.append(soup2.select(".FullRecTitle")[0].value.text.encode('utf-8'))
                abstracts.append(soup2.find(text="Abstract:").parent.parent.text.encode('utf-8'))
                categories.append(soup2.find(text="Web of Science Categories:").next_element.encode('utf-8'))

            time.sleep(random.randint(1, randInt))

    # process the results
    result = []
    for i in xrange(len(dois)):
        tmp = [dois[i], titles[i], abstracts[i], categories[i]]
        result.append(tmp)

    # gather the results
    final = comm.gather(result, root=0)

    comm.barrier()
    p_stop = MPI.Wtime()

    # compile and save
    if rank == 0:
        p_result = list(itertools.chain.from_iterable(final))

        # serial
        s_start = time.time()
        s_result = serial(topic)
        s_stop = time.time()

        print "Serial Time: %f secs" % (s_stop - s_start)
        print "Parallel Time: %f secs" % (p_stop - p_start)

        # check
        errors = 0
        for i in xrange(len(p_result)):
            if p_result[i][0] != s_result[i][0]: 
                print "ERROR: %d (%s, %s)" % (i, p_result[i][0], s_result[i][0])
            elif p_result[i][1] != s_result[i][1]: 
                print "ERROR: %d (%s, %s)" % (i, p_result[i][1], s_result[i][1])
            elif p_result[i][2] != s_result[i][2]: 
                print "ERROR: %d (%s, %s)" % (i, p_result[i][2], s_result[i][2])
            elif p_result[i][3] != s_result[i][3]:
                print "ERROR: %d (%s, %s)" % (i, p_result[i][3], s_result[i][3])
        print "TOTAL ERRORS: %d" % errors

        # save
        save(p_result, topic)
    

if __name__ == '__main__':
    if len( sys.argv ) < 3:
        print 'Usage: ' + sys.argv[0] + ' <topic> [-m or -s]'
        sys.exit(0)
    else:
        if (sys.argv[2] == '-m'):
            masterslave(sys.argv[1])
        else:
            scattergather(sys.argv[1])
