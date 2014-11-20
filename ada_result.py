import numpy as np
import weaklearners as weak
import matplotlib.pyplot as plt
#savepath = "/home/ubuntu/saveddata/"
savepath = "./results/"
## all the features are stored locally
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
	alphas = np.loadtxt('results/alphas.csv', delimiter=',')
	features = np.loadtxt('results/selected_features.csv',  delimiter=',',dtype=int)
	weights = np.loadtxt('results/weights.csv',  delimiter=',')
	print alphas.shape
	print features.shape
	print weights.shape
	displayTopTen(features)
	# faces = weak.Images("../newface"+str(16)+"/",isface=True, samplesize=6000, imgwidth=16)
	# nonfaces = weak.Images("../nonface"+str(16)+"/",isface=False, samplesize=6000, imgwidth=16)

	# face_table = faces.GetTables()
	# nonface_table = nonface.GetTables()

	# face_score_Gen = weak.Scores(face_table)
	# nonface_score_Gen = weak.Scores(nonface_table)


#At steps T=0, 10, 50, 100 respectively, plot the curve for the errors of  top 1000 weak classifiers among the pool of weak classifiers in increasing order.  
#                        Compare these four curves  and see how many of the weak classifiers have errors close to 1/2;
#             iii)  Plot the histograms of the positive and negative populations over the F(x) axis, for T=10, 50, 100 respectively.
#                     From the three histograms, you plot their corresponding ROC curves.