import os
import sys
from sys import argv
import numpy as np
from abstract import Abstract

if __name__ == '__main__':
    script, filename = argv
    os.system("python process.py "+filename)
    print "What article are you searching for?",
    article = raw_input()
    keywords = article.split()
