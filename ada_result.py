import numpy as np
import weaklearners as weak
import matplotlib.pyplot as plt
import ada_train
#savepath = "/home/ubuntu/saveddata/"
savepath = "../saveddata/"
from multiprocessing import Process, Queue

Q = Queue()

## global arguments




class SortByError(ada_train.ProcessWorker):
	def __init__(self, ftype, scores, labels, weights):
		ada_train.__init__(self,ftype, scores, labels, weights)
	def run(self):
		##(error, ftype, row)
		results = []
		pool = range(self.scores.shape[0])
		for row in pool:
			threshold, error, flag = self.FindFeatureError(row)
			#results.append((error, self.ftype, row))
			results.append(error)
		Q.put(results)

def dump_queue(queue):
	l = []
	while not queue.empty():
		l.append(queue.get())
	return l

## input: scores and labels are both lists of numpy arrays
## output: sorted top 1000 error
def SortFindMinError(scores, labels, weights):
	pros = []
	for i in range(1,7):
		p = SortByError(i, scores[i], labels[i], weights)
		pros.append(p)
		p.start()
	for p in pros:
		p.join()
	results = dump_queue(Q)
	##(error, ftype, row)
	## sort the results according to the error
	results.sort()
	return np.array(results[0:1000])


class Results():
	"""docstring for Results"""
	def __init__(self):
		self.scores = []
		self.labels = []
		self.features = []
		for i in range(1,7):
			score = np.load(savepath+'scores_feature_type'+str(i)+'.npy')
			label = np.load(savepath+'scores_labels_type'+str(i)+'.npy')
			feature = np.loadtxt(savepath+'feature_type_'+str(i)+'.csv', delimiter=',',dtype=int)
			self.scores.append(score)
			self.labels.append(label)
			self.features.append(feature)
		self.alphas = np.loadtxt('results/alphas.csv', delimiter=',')
		self.selected_features = np.loadtxt('results/selected_features.csv',  delimiter=',',dtype=int)
		self.weights = np.loadtxt('results/weights.csv', delimiter=',')

	#At steps T=0, 10, 50, 100 respectively, plot the curve for the errors of  top 1000 weak classifiers among the pool of weak classifiers in increasing order. 
	def ErrorCurve(self):
		for i in range(10):
			np.save('results/error_curve_iteraton0', SortFindMinError(self.scores, self.labels, self.weights[i]))
		#np.save('results/error_curve_iteraton10', SortFindMinError(self.scores, self.labels, self.weights[9]))
		#np.save('results/error_curve_iteraton50', SortFindMinError(self.scores, self.labels, self.weights[49]))
		#np.save('results/error_curve_iteraton100', SortFindMinError(self.scores, self.labels, self.weights[99]))
	#iii)  Plot the histograms of the positive and negative populations over the F(x) axis, for T=10, 50, 100 respectively.
	#From the three histograms, you plot their corresponding ROC curves.
	#def Populations(self):

		
			
		


	## read the scores and labels
	## read the weights
	## call the SortFindMinError to get the results


## now given an index, return the cooresponding feature
def getFeature(ftype, index):
	feature_table = np.loadtxt('./saveddata/feature_type_'+str(int(ftype))+'.csv', delimiter=',',dtype=int)
	print feature_table[index]
	return feature_table[index]




# Display the best ten features as images (after boosting);
def displayTopTen(features):
	fimg_Gen = weak.Features(16)
	for row in range(10):
		ff = getFeature(features[row, 0], features[row, 1])
		featureimg = fimg_Gen.GetFeatureImg(ff, features[row, 0])
		print "feature image,\n", featureimg
		ax = plt.subplot(2,5,row)
		ax.axis('off')
		imgplot = plt.imshow(featureimg, cmap=plt.cm.gray, vmax=1, vmin=-2)
		imgplot.set_interpolation('nearest')
	plt.savefig('results/topten.png', dpi=150)


if __name__ == "__main__":
	# load the results
	r = Results()
	r.ErrorCurve()

	# faces = weak.Images("../newface"+str(16)+"/",isface=True, samplesize=6000, imgwidth=16)
	# nonfaces = weak.Images("../nonface"+str(16)+"/",isface=False, samplesize=6000, imgwidth=16)

	# face_table = faces.GetTables()
	# nonface_table = nonface.GetTables()

	# face_score_Gen = weak.Scores(face_table)
	# nonface_score_Gen = weak.Scores(nonface_table)