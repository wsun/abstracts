from mrjob.job import MRJob
from collections import defaultdict
import math

class tfidfwfFrequency(MRJob):

	def mapper_init(self):
		self.words = defaultdict(float)

	def mapper(self,key,line):
	    abstext = line.split()
	    for word in abstext:
		    word = word.join([c for c in word if c.isalnum()])
		    self.words[word] += 1

	def mapper_final(self):
		for word, count in self.words.items():
			yield word, count

	def reducer(self,word,counts):
		yield word, sum(counts)

if __name__ == '__main__':
	tfidfwfFrequency.run()
	

    for abstract in abstracts:
        # create dict of word frequency using TF-IDF
        normwf = defaultdict(int)
        for word, freq in abs.Get('wf'):
            normwf[word].append(freq*math.log(float(termdoc[word]/len(termdoc))))
        abs.Set('tfidfwf', normwf)