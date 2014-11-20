#import h5py
import numpy as np
from numpy import matrix
import threading


## the mechanism of process--class. values cannot be accessed after join
#http://stackoverflow.com/questions/7545385/python-class-inheriting-multiprocessing-trouble-with-accessing-class-members
from multiprocessing import Process, Queue
import timeit
from mapreduce import SimpleMapReduce
import sys

savepath = "../saveddata/" 

lock = threading.Lock()
Q = Queue()
class ProcessWorker(Process):
	"""
	This class runs as a separate process to execute worker's commands in parallel
	Once launched, it remains running, monitoring the task queue, until "None" is sent
	"""
	def __init__(self, ftype, scores, labels, weights):
		Process.__init__(self)
		# load the table
		assert(scores.shape==labels.shape)
		self.ftype = ftype
		#self.f = h5py.File(savepath+'scores_feature_type'+str(ftype)+'.hdf5','r')
		self.scores = scores
		self.labels = labels
		self.weights = weights
		self.mapResult = []
		self.num_sample = scores.shape[1]
		return
	#
	# def changeweights(self, newweights):
	# 	self.weights = newweights


	# def fetchResult(self):
	# 	print "fetching results for type",self.ftype
	# 	return self.min_error, self.min_row

	# def firstFetch(self):
	# 	# first attempt can be avoided
	# 	# see http://cplusadd.blogspot.com/2013/04/why-class-balancing-happens.html
	# 	self.min_threshold, self.min_error, self.min_flag = self.FindFeatureError(0)
	# 	self.min_row = 0
	# 	return self.featureChosen()


	# get the weighted error for the i-th feature
	# return decision threshold, error, decision flag
	#@staticmethod
	
	# return decision threshold, error, decision flag
	def FindFeatureError(self,row):
			# understand the threshold
			# http://stackoverflow.com/questions/9777282/the-best-way-to-calculate-the-best-threshold-with-p-viola-m-jones-framework
			above_is_positive = True
			## initialize the threshold so that everyone is determined to be positive
			assert(np.fabs(sum(self.weights)-1.0)<0.000001)
			error = sum(self.weights[~self.labels[row,:]])
			maxerror = [-1, error]
			minerror = [-1, error]
			for j in range(self.labels.shape[1]):
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
				lock.acquire()
				self.silly_count += 1
				lock.release()
				#print "Silly Threshold Found. Min Error,", minerror[1]
			# return decision threshold, error, decision flag
			return minerror[0], minerror[1], above_is_positive
	# one thread responsible for multiple rows
	def MapFind(self, rowlist, mid):
		lock.acquire()
		lock.release()
		minError = 1
		minRow = None
		i = 0
		for row in rowlist:
			error_infor = self.FindFeatureError(row)
			if error_infor[1]<minError:
				minError = error_infor[1]
				minResult = error_infor
				minRow = row
		self.mapResult.append((minResult,minRow))
		print "fType"+str(self.ftype)+", mID"+str(mid)+" finished."
		print "Min Feature:", minRow

	def Reduce(self):
		minError = 1
		minRow = None
		result = None
		for error_infor,row in self.mapResult:
			if error_infor[1]<minError:
				minError =error_infor[1]
				result = error_infor
				minRow = row
		
		self.mapResult = []
		self.min_threshold = result[0]
		self.min_error = result[1]
		self.min_flag = result[2]
		self.min_row = minRow
		Q.put((self.ftype, self.min_row, self.min_error, self.min_threshold, self.min_flag))
		print "fType"+str(self.ftype)+" reduced! Min Error", self.min_error
		print 'feature withMinimum error', minRow
		print "silly Count:", self.silly_count, 'out of', self.scores.shape[0]

	def run(self):
		## findMinError
		"""
		Overloaded function provided by multiprocessing.Process.  Called upon start() signal
		"""
		self.silly_count = 0
		print 'Starting Process type', self.ftype
		print "Total number of samples,",self.num_sample
		self.min_error = 1
		# it = 0
		#self.threadnum = min(500, len(self.pool))
		threadnum = 15
		self.pool = range(self.scores.shape[0])
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



class FeaturePool():
	def __init__(self, ftypeMax, num_feature, num_sample):
		self.num_sample = num_sample
		self.num_feature = num_feature
		self.ftypeMax = ftypeMax
		fNum = num_feature/ftypeMax

		# initialize masks and positions to record selected ones
		self.mask = np.ones((ftypeMax, fNum), dtype=bool)
		self.index = np.array(range(fNum))


		
		## initialize weights
		self.weights = np.empty(num_sample)
		self.weights.fill(1.0/num_sample)

		self.scores = []
		self.labels = []
		for i in range(ftypeMax):
			score = np.load(savepath+'scores_feature_type'+str(i+1)+'.npy')
			label = np.load(savepath+'scores_labels_type'+str(i+1)+'.npy')
			self.scores.append(score)
			self.labels.append(label)

		self.alphas = []
		self.selected = []
		self.hist_weigts = [self.weights]


	# to-do
	# 0. reduce work, find minimum
	# 1. remove the feature from the pool
	# 2. update the weights
	# 3. return the alpha and updated weights to inform others
	def ReduceWorkers(self):
		self.min_error = 1
		while not Q.empty():
			item = Q.get()
			if item[2] < self.min_error:
				self.min_type = item[0]
				# getting the row number of the selected feature is tricky
				self.min_row = self.index[self.mask[item[0]-1,:]][item[1]]
				self.min_error = item[2]
				self.min_threshold = item[3]
				self.min_flag = item[4]
			#(self.ftype, self.min_row, self.min_error, self.min_threshold, self.min_flag)
		print "Min Error found:", self.min_error
		print "fType:", self.min_type, "Position:", self.min_row
		# calculate the alpha
		alpha = 0.5*np.log((1-self.min_error)/self.min_error)
		## store the selected feature
		self.alphas.append(alpha)
		self.selected.append((self.min_type, self.min_row))
		## mask the selected feature 
		print "Masking ftype"+str(self.min_type-1)+" row"+str(self.min_row)
		self.mask[self.min_type-1, self.min_row] = False
		
		# update the weights of the data points
		for i in range(self.num_sample):
			above =  self.scores[self.min_type-1][self.min_row, i] > self.min_threshold
			# if above threshold and above is positive
			cur_decision = 1 if (above == self.min_flag) else -1
			cur_label = 1 if self.labels[self.min_type-1][self.min_row, i] else -1
			self.weights[i] = self.weights[i]*np.exp(-cur_label*alpha*cur_decision)
		self.weights = self.weights/sum(self.weights)
		self.hist_weigts.append(self.weights)
		
	def SaveResults(self):
		np.savetxt('results/alphas.csv',np.array(self.alphas), fmt='%d', delimiter=',')
		np.savetxt('results/selected_features.csv', np.array(self.selected),fmt='%d', delimiter=',')
		np.savetxt('results/weights.csv', np.array(self.hist_weigts), delimiter=',')


	def GetFeaturePool(self, ftype):
		return self.scores[ftype-1][self.mask[ftype-1,:],:]

	def GetLabels(self,ftype):
		return self.labels[ftype-1][self.mask[ftype-1,:],:]

	def GetWeights(self):
		return self.weights

def test():
	#ftypeMax, num_feature, num_sample)
	fpool = FeaturePool(1, 1000, 6000*2)
	p = ProcessWorker(1, fpool.GetFeaturePool(1), fpool.GetLabels(1), fpool.GetWeights())
	p.MapFind([331], 0)

	print "\nIteration"


if __name__ == "__main__":
	start = timeit.default_timer()
	T = 25
	numfType = 6
	#ftypeMax, num_feature, num_sample)
	fpool = FeaturePool(6, 6000, 6000*2)

	t = 0
	while t<T: 
		t += 1
		pros = []
		for i in range(1,numfType+1):
			#ftype, scores, labels, weights
			p = ProcessWorker(i, fpool.GetFeaturePool(i), fpool.GetLabels(i), fpool.GetWeights())
			p.start()
			pros.append(p)
		for p in pros:
			p.join()
		## to do, reduce work, find alpha, store alpha and min position,
		## and change weights
		print "\nIteration",t,"Finished!"
		fpool.ReduceWorkers()
	fpool.SaveResults()
	stop = timeit.default_timer()
	print "Time Used,", round(stop - start, 4)

