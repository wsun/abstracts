from mrjob.job import MRJob
from collections import defaultdict

class wordFrequency(MRJob):

	def mapper_init(self):
		self.words = defaultdict(int)

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
	wordFrequency.run()