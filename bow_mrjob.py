from mrjob.job import MRJob
from collections import defaultdict

class bowFrequency(MRJob):

	def mapper_init(self):
		self.phrases = defaultdict(int)

	def mapper(self,key,line):
	    abstext = line.split()
	    for i in range(len(abstext)-bagsize+1):
            wordbag = abstext[i:i+bagsize].sort()
            self.phrases[wordbag] += 1

	def mapper_final(self):
		for phrase, count in self.words.items():
			yield phrase, count

	def reducer(self,word,counts):
		yield phrase, sum(counts)

if __name__ == '__main__':
	bowFrequency.run()