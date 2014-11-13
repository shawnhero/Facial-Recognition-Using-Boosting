import h5py
import numpy as np
from numpy import matrix
import threading
from multiprocessing import Process

class ProcessWorker(Process):
    """
    This class runs as a separate process to execute worker's commands in parallel
    Once launched, it remains running, monitoring the task queue, until "None" is sent
    """
    def __init__(self, scores, labels):
		Process.__init__(self)
		self.num_sample = scores.shape[0]
		self.num_feature = scores.shape[1]
		self.scores = scores
		self.labels = labels
		self.pool = range(num_feature)
		return
	def changeweights(self, newweights):
		self.weights = newweights
	def featureChosen(self, col):
		self.pool.remove(col)
	def fetchResult(self):
		return minerror, index

    def run(self):
		## findMinError
		"""
		Overloaded function provided by multiprocessing.Process.  Called upon start() signal
		"""
		minerror = 1
		index = 0
		for col in self.pool:
			cur_error = self.FindFeatureError(col)
			if cur_error <  minerror:
				minerror = cur_error
				index = col
		self.minerror = minerror
		self.minindex = index
		return
	# get the weighted error for the i-th classifier
	def FindFeatureError(self,col):
		# find the threshold of the col-th classifier
		# find positive and negative mean
		pmask = np.asarray(self.labels, dtype=bool)
		nmask = np.asarray(1-self.labels, dtype=bool)
		pmean = np.mean(self.scores[pmask, col])
		nmean = np.mean(self.scores[nmask, col])
		# find the threshold between the two means
		step = np.fabs(pmean-nmean)/100
		thresholdlist = [(pmean if pmean<nmean else nmean)+step*i for i in range(101)]
		errors = []
		for t in thresholdlist:
			flag_below = True if pmean<nmean else False
			decisions = np.asarray([flag_below if self.scores[j,col]<=t else 1- flag_below for j in range(self.num_sample)], dtype=bool)
			error = matrix(np.fabs(decisions - self.labels).reshape(1, self.num_sample))*matrix(self.weights.reshape(self.num_sample,1))
			errors.append(error[0,0])
		if 1-max(errors) < min(errors):
			print "Need a flip over! Didn't expect that could happen.."
		threshold = thresholdlist[errors.index(min(errors))]
		print "pmean", pmean
		print "nmean", nmean
		print "threshold:", threshold
		#
		# we don't need decide for this moment
		#decisions = np.asarray([flag_below if self.scores[j,col]<=threshold else 1-flag_below for j in range(self.num_sample)], dtype=bool)
		return min(errors)

class adaboosting():
	# weights = None
	# num_sample = None
	# num_classifier = None
	def __init__(self, scores, labels, T):
		assert (scores.shape[0]==labels.shape[0])
		#assert (labels.shape[1]==1)
		self.scores = scores
		self.labels = labels
		self.num_sample = labels.shape[0]
		self.num_classifier = scores.shape[1]
		self.T = T
		self.D = np.empty(self.num_sample)
		self.D.fill(1.0/self.num_sample)
		# do something here
	

	def my_func(self, classifier):
		return np.fabs(classifier - self.labels)

	def weightedErrors(self, classifier_list):
		errors = np.apply_along_axis(self.my_func, 0, classifier_list)
		#error = numpy.fabs(classifiers[:][i]-labels)
		return np.asarray(matrix(self.D.reshape(1, self.num_sample))*matrix(errors))

	def boosting(self):
		step = 0
		mask = np.ones(self.num_classifier, dtype=bool)
		indexes = np.asarray(range(self.scores.shape[1]))
		alphas = []
		while step <= self.T:
			step += 1
			cur_results = np.asarray([self.findweakresults(indexes[mask][k]) for k in range(sum(mask))]).transpose() 
			errors = self.weightedErrors(cur_results)
			eindex = np.argmin(errors)#errors.index(minerror)
			minerror = errors[0,eindex]

			## debug use:
			print "min error,", minerror

			## debug end
			index = indexes[mask][eindex]
			mask[index] = False

			alpha = 0.5*np.log((1-minerror)/minerror)
			alphas.append(alpha)
			# update the weights of data points
			D_next = [self.D[i]*np.exp(-self.labels[i]*alpha*self.scores[i][index])for i in range(self.num_sample)]
			self.D = D_next/sum(D_next)
		return mask, alphas

def test():
	labels = [1,1,1,0,0,0]
	classifiers = [[1,1,0,1,0,0], [1,1,1,1,0,0]]
	ada = adaboosting(np.asarray(classifiers).transpose(), np.asarray(labels), 1)
	ada.boosting()

	return 1

if __name__ == "__main__":
	test()