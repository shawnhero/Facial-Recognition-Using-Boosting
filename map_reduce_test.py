## MapReduce Test
import multiprocessing
import threading
from mapreduce import SimpleMapReduce
import numpy as np
import h5py
lock = threading.Lock()


class test():
	#@staticmethod
	def __init__(self):
		savepath = "/Users/Shawn/cs276/data/"
		self.f = h5py.File(savepath+'scores_feature_type1.hdf5','r')
	def map(self,rowlist):
		rowmin = 10
		self.prints()
		for row in rowlist:
			if row < rowmin:
				rowmin = row
		lock.acquire()
		print "map called,:", rowlist
		lock.release()
		return [(0, rowmin)]
	#@staticmethod
	def reduce(self,item):
		lock.acquire()
		print "reduce called", item
		lock.release()
		key, mins = item
		rowmin = 10
		for ele in mins:
			if ele<rowmin:
				rowmin =ele
		return (key, rowmin)
	def start(self):
		mapper = SimpleMapReduce(self.map, self.reduce, num_workers=5)
		inputlist = [[1,2,3,4,4,4,5,5,5,5,5,6,6,6,7,9,3], [3,4,5],[6,7,8],[9,10,11]]
		result = mapper(inputlist)
		print result
	def prints(self):
		self.scores = self.f['type1']
		self.value = self.scores[100,100]
		print self.value

if __name__ == '__main__':
	t = test()
	t.start()
	# mapper = SimpleMapReduce(t.map, t.reduce, num_workers=5)
	# inputlist = [[1,2,3,4,4,4,5,5,5,5,5,6,6,6,7,9,3], [3,4,5],[6,7,8],[9,10,11]]
	# result = mapper(inputlist)
	# print result

