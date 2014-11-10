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
		while step < self.T:
			step += 1
			errors = self.weightedErrors(self.classifiers[:,mask])
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
	select, alphas = ada.boosting()
	return 1

if __name__ == "__main__":
	test()