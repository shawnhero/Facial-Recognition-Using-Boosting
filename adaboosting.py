import numpy as np
from numpy import matrix
class adaboosting():
	# weights = None
	# num_sample = None
	# num_classifier = None
	def __init__(self, classifiers, labels, T):
		assert (classifiers.shape[0]==labels.shape[0])
		#assert (labels.shape[1]==1)
		self.classifiers = classifiers
		self.labels = labels
		self.num_sample = labels.shape[0]
		self.num_classifier = classifiers.shape[1]
		self.T = T
		self.D = np.empty(self.num_sample)
		self.D.fill(1.0/self.num_sample)
		# do something here
	# get the weighted error for the i-th classifier
	def findweakresults(self,col):
		# find the threshold of the col-th classifier
		# find positive and negative mean
		pmask = np.asarray(self.labels, dtype=bool)
		nmask = np.asarray(1-self.labels, dtype=bool)
		pmean = np.mean(self.classifiers[pmask, col])
		nmean = np.mean(self.classifiers[nmask, col])
		# find the threshold between the two means
		step = np.fabs(pmean-nmean)/100
		thresholdlist = [(pmean if pmean<nmean else nmean)+step*i for i in range(101)]
		errors = []
		for t in thresholdlist:
			decisions = np.asarray([0 if self.classifiers[j,col]<=t else 1 for j in range(self.num_sample)])
			if pmean < nmean:
				decisions = 1 - decisions
			error = matrix(np.fabs(decisions - self.labels).reshape(1, self.num_sample))*matrix(self.D.reshape(self.num_sample,1))
			errors.append(error[0,0])
		if 1-max(errors) < min(errors):
			print "Need a flip over! Didn't expect that could happen.."
		threshold = thresholdlist[errors.index(min(errors))]
		print "pmean", pmean
		print "nmean", nmean
		print "threshold:", threshold
		# return our decision
		return np.asarray([0 if self.classifiers[j,col]<=threshold else 1 for j in range(self.num_sample)])

	def my_func(self, classifier):
		return np.fabs(classifier - self.labels)

	def weightedErrors(self, classifier_list):
		errors = np.apply_along_axis(self.my_func, 0, classifier_list)
		#error = numpy.fabs(classifiers[:][i]-labels)
		return np.asarray(matrix(self.D.reshape(1, self.num_sample))*matrix(errors))

	def boosting(self):
		step = 0
		mask = np.ones(self.num_classifier, dtype=bool)
		indexes = np.asarray(range(self.classifiers.shape[1]))
		alphas = []
		while step <= self.T:
			step += 1
			cur_results = np.asarray([self.findweakresults(indexes[mask][k]) for k in range(sum(mask))]).transpose() 
			errors = self.weightedErrors(cur_results)
			eindex = np.argmin(errors)#errors.index(minerror)
			minerror = errors[0,eindex]
			index = indexes[mask][eindex]
			mask[index] = False

			alpha = 0.5*np.log((1-minerror)/minerror)
			alphas.append(alpha)
			# update the weights of data points
			D_next = [self.D[i]*np.exp(-self.labels[i]*alpha*self.classifiers[i][index])for i in range(self.num_sample)]
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