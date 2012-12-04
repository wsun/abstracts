# CS205 Final Project
# Janet Song and Will Sun
#
# Scraper for Web of Science using MPI
# 
# Use provided runscript

from mpi4py import MPI
import time
import random
import urllib2
from bs4 import BeautifulSoup

# base URLs
topics = {
            'biology': 'http://apps.webofknowledge.com/summary.do?product=WOS&search_mode=GeneralSearch&qid=3&SID=1B27ElmfB5jcB6kN@5H&&page=',
            'chemistry': 'http://apps.webofknowledge.com/summary.do?product=WOS&search_mode=GeneralSearch&qid=4&SID=1B27ElmfB5jcB6kN@5H&&page=',
            'physics': 'http://apps.webofknowledge.com/summary.do?product=WOS&search_mode=GeneralSearch&qid=5&SID=1B27ElmfB5jcB6kN@5H&&page=',
            'medicine': 'http://apps.webofknowledge.com/summary.do?product=WOS&search_mode=GeneralSearch&qid=6&SID=1B27ElmfB5jcB6kN@5H&&page=',
            'ecology': 'http://apps.webofknowledge.com/summary.do?product=WOS&search_mode=GeneralSearch&qid=7&SID=1B27ElmfB5jcB6kN@5H&&page='
         }

agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.95 Safari/537.11',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/536.26.17 (KHTML, like Gecko) Version/6.0.2 Safari/536.26.17',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.4 (KHTML, like Gecko) Chrome/22.0.1229.94 Safari/537.4',
         ]

# maximum sleep
randInt = 3


### DEFINE
topic = 'biology'
page = 1
url = topics[topic] + str(page)

# make the request
r = urllib2.Request(url)
r.add_header('User-Agent', random.choice(agents))
response = urllib2.urlopen(r)
soup = BeautifulSoup(response, 'lxml')

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

    # capture the DOI
    dois.append(article.find(text="DOI: ").next_element.text)

    # capture the title
    titles.append(article.select(".smallV110")[0].value.text)

# navigate to get abstracts and categories
for l in links:
    time.sleep(random.random())

    r2 = urllib2.Request(l)
    r2.add_header('User-Agent', random.choice(agents))
    response2 = urllib2.urlopen(r2)
    soup2 = BeautifulSoup(response, 'lxml')

    abstracts.append(soup2.find(text="Abstract:").parent.parent.text)
    categories.append(soup2.find(text="Web of Science Categories:").next_element)

    time.sleep(random.randint(1, randInt))


