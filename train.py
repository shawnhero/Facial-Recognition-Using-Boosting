import adaboosting as ada
import weaklearners as weak
import numpy as np
import matplotlib.pyplot as plt
import timeit

from multiprocessing import Process
import threading

savepath = "/Users/Shawn/cs276/"

class ProcessWorker(Process):
    """
    This class runs as a separate process to execute worker's commands in parallel
    Once launched, it remains running, monitoring the task queue, until "None" is sent
    """

    def __init__(self, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        return

    def run(self):
        """
        Overloaded function provided by multiprocessing.Process.  Called upon start() signal
        """
def train(test=True):
	## load the data
	print "Loading the score tables..."
	facescores = np.loadtxt("./data/face_scores.csv", delimiter=',')
	print "Face scores loaded!"
	nonfacescores = np.loadtxt("./data/nonface_scores.csv", delimiter=',')
	print "Nonface scores loaded!"
	allscores = np.concatenate((facescores, nonfacescores), axis=0)
	labels = np.concatenate([np.ones(facescores.shape[0]), np.zeros(nonfacescores.shape[0])  ])

	if test:
		model = ada.adaboosting(allscores, labels, 1)
		mask, alphas = model.boosting()
		print "alphas: ", alphas
		minindex = np.argmin(mask)
		print "mask index: ", minindex
		minerr_feature, ftype = getFeature(minindex, 12000)
		print "ftype: ", ftype
		
		# here we run a test and demonstrate the first feature we select after first run
		f = weak.Features(24)
		featureimg = f.GetFeatureImg(minerr_feature, ftype)
		imgplot = plt.imshow(featureimg, cmap=plt.cm.gray, vmax=1, vmin=-2)
		imgplot.set_interpolation('nearest')
		plt.show()


def getFeature(i, numfeatures):
	numfeatures = (numfeatures/6)*6
	numsamples_per_feature = numfeatures/6
	position = i/numsamples_per_feature + 1
	index = i%numsamples_per_feature
	feature_table = np.loadtxt('./data/feature_type_'+str(position)+'.csv', delimiter=',')
	return feature_table[index], position


if __name__ == "__main__":
	start = timeit.default_timer()
	train(test=True)
	stop = timeit.default_timer()
	print "Time Used,", round(stop - start, 4)



# np.loadtxt("face_scores.csv", delimiter=',')
# np.save('face_scores.npy', a)
#  X = np.load('face_scores.npy', mmap_mode='c')
#  np.loadtxt("nonface_scores.csv", delimiter=',')

# pd.read_csv("nonface_scores.csv", delimiter=",")
