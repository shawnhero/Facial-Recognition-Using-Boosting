import h5py
import numpy as np
from numpy import matrix
import threading
from multiprocessing import Process
import timeit
from mapreduce import SimpleMapReduce
import sys

savepath = "/Users/Shawn/cs276/data/"
lock = threading.Lock()
class ProcessWorker(Process):
	"""
	This class runs as a separate process to execute worker's commands in parallel
	Once launched, it remains running, monitoring the task queue, until "None" is sent
	"""
	def __init__(self, ftype, numfaces, numnonfaces):
		Process.__init__(self)
		# load the table
		self.ftype = ftype
		self.f = h5py.File(savepath+'scores_feature_type'+str(ftype)+'.hdf5','r')
		self.scores = self.f['type'+str(ftype)]
		self.labels = self.f['labels']
		self.num_sample = self.scores.shape[1]
		assert(self.num_sample==numfaces+numnonfaces)
		self.num_feature = self.scores.shape[0]

		# initialize the pools
		self.pool = range(self.num_feature)
		self.mapResult = []
		return
	def changeweights(self, newweights):
		self.weights = newweights

	# to-do
	# 1. remove the feature from the pool
	# 2. update the weights
	# 3. return the alpha and updated weights to inform others
	def featureChosen(self):
		print "Min error found:", self.min_error
		print "postion:", self.min_row
		# calculate the alpha
		self.alpha = 0.5*np.log((1-self.min_error)/self.min_error)
		
		# update the weights of the data points
		for i in range(self.num_sample):
			above =  self.scores[self.min_row, i] > self.min_threshold
			# if above threshold and above is positive
			cur_decision = 1 if (above == self.min_flag) else -1
			cur_label = 1 if self.labels[self.min_row, i] else -1
			self.weights[i] = self.weights[i]*np.exp(-cur_label*self.alpha*cur_decision)
		self.weights = self.weights/sum(self.weights)
		try:
			self.pool.remove(self.min_row)
		except ValueError:
			print 'min_row',self.min_row,'not in the list'
			sys.exit()
		return self.alpha, self.weights


	def fetchResult(self):
		print "fetching results for type",self.ftype
		return self.min_error, self.min_row
	def firstFetch(self):
		# first attempt can be avoided
		# see http://cplusadd.blogspot.com/2013/04/why-class-balancing-happens.html
		self.min_threshold, self.min_error, self.min_flag = self.FindFeatureError(0)
		self.min_row = 0
		return self.featureChosen()


	# get the weighted error for the i-th feature
	# return decision threshold, error, decision flag
	#@staticmethod
	
	# return decision threshold, error, decision flag
	def FindFeatureError(self,row):
			# understand the threshold
			# http://stackoverflow.com/questions/9777282/the-best-way-to-calculate-the-best-threshold-with-p-viola-m-jones-framework
			above_is_positive = True
			## initialize the threshold so that everyone is determined to be positive
			error = sum(self.weights[~self.labels[row,:]])
			maxerror = [-1, error]
			minerror = [-1, error]
			for j in range(self.num_sample):
				# for those<=j, decide as negative
				# for those>j, decide as positive
				if self.labels[row,j]:
					error += self.weights[j]
				else:
					error -= self.weights[j]
				if error>maxerror[1]:
					maxerror[0] = j
					maxerror[1] = error
				if error<minerror[1]:
					minerror[0] = j
					minerror[1] = error
			if 1- maxerror[1] < minerror[1]:
				# we need a flip over
				above_is_positive = False
				minerror = maxerror
				minerror[1] = 1- minerror[1]
			if minerror[0]==-1 or minerror[0]==self.num_sample-1:
				print "Silly Threshold Found. Min Error,", minerror[1]
				# print "Decision Threshold at,", minerror[0]
				# print "Above is "+("Positive" if above_is_positive else "Negative")
			else:
				pass
				# print 'Decision found for','type'+str(self.ftype), 'row'+str(row)
			# return decision threshold, error, decision flag
			return minerror[0], minerror[1], above_is_positive
	# one thread responsible for multiple rows
	def MapFind(self, rowlist, mid):
		lock.acquire()
		print "fType"+str(self.ftype)+", mID"+str(mid)+". Mapping:", rowlist[0],"to",rowlist[len(rowlist)-1], "Number,", len(rowlist)
		print 
		lock.release()
		minError = 1
		minRow = None
		i = 0
		for row in rowlist:
			i += 1
			if i % 3 ==0:
				lock.acquire()
				print "fType"+str(self.ftype)+", mID"+str(mid)+", {0:.1%}".format(1.0*i/len(rowlist)), "done.."
				lock.release()
			error_infor = self.FindFeatureError(row)
			if error_infor[1]<minError:
				minError = error_infor[1]
				minResult = error_infor
				minRow = row
		self.mapResult.append((minResult,minRow))
		print "fType"+str(self.ftype)+", mID"+str(mid)+" finished."

	def Reduce(self):
		minError = 1
		minRow = None
		result = None
		for error_infor,row in self.mapResult:
			if error_infor[1]<minError:
				minError =error_infor[1]
				result = error_infor
				minRow = row
		print "fType"+str(self.ftype)+' reduced: feature withMinimum error', minRow
		self.mapResult = []
		self.min_threshold = result[0]
		self.min_error = result[1]
		self.min_flag = result[2]
		self.min_row = minRow
		print "fType"+str(self.ftype)+" reduced! Min Error", self.min_error

	def run(self):
		## findMinError
		"""
		Overloaded function provided by multiprocessing.Process.  Called upon start() signal
		"""
		self.count = 0
		print 'Starting Process type', self.ftype
		self.min_error = 1
		# it = 0
		#self.threadnum = min(500, len(self.pool))
		threadnum = 10
		rows = len(self.pool)/threadnum
		list_rowlists = [self.pool[x:x+rows] for x in xrange(0, len(self.pool), rows)]
		threadnum = len(list_rowlists)
		# use 10 thread for each process to find the min
		threads = []
		for i in range(threadnum):
			##self.MapFind(list_rowlists[i])
			t = threading.Thread(target=self.MapFind, args=(list_rowlists[i],i,))
			threads.append(t)
			t.start()
		for t in threads:
			t.join()
		self.Reduce()

		
		# it += 1
		# if it%10==0:
		# 	print 'type'+str(self.ftype),"{0:.1%}".format(float(it)/len(self.pool)), ' search completed'
		return


if __name__ == "__main__":
	start = timeit.default_timer()
	T = 3
	t = 0
	# initialize the weights
	num_sample = 6000*2
	D = np.empty(num_sample)
	D.fill(1.0/num_sample)

	# initialize the subprocesses
	# load the tables
	pros = []
	for i in range(1,3):
		p = ProcessWorker(i, 6000, 6000)
		# initialize the weights
		p.changeweights(D)
		pros.append(p)
	# feature positions
	fpos = []
	while t<T: 
		t += 1
		if t==1:
			# first fetch
			alpha, D = pros[0].firstFetch()
			minpos = (0,0)
		else:
			for p in pros:
				p.start()
			for p in pros:
				p.join()
			# find the min error
			minerror = 1
			minpos = 0
			for i in range(len(pros)):
				error, row = pros[i].fetchResult()
				if error < minerror:
					minerror = error
					minpos = (i, row)
			print 'minpos',minpos
			# remove the chosen feature from the pool
			alpha, D = pros[minpos[0]].featureChosen()
		# store the chosen feature's position
		fpos.append(minpos)
		
		# update the weights of the data points
		for i in pros:
			p.changeweights(D)
	print "chosen features:", fpos
	stop = timeit.default_timer()
	print "Time Used,", round(stop - start, 4)


def test():
	num_sample = 6000*2
	D = np.empty(num_sample)
	D.fill(1.0/num_sample)

	p = ProcessWorker(1, 6000, 6000)
	p.changeweights(D)
	alpha, D = p.firstFetch()
	error, row = p.fetchResult()
	p.run()
	alpha, D = p.featureChosen()
	p.changeweights(D)
	p.run()
