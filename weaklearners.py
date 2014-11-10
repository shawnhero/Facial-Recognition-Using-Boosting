#Design a few types of features as we discussed in class.
# For each feature plot the histograms for the positive and negative populations. Determine the threshold.
#[note that as the samples change their weight over time, the histogram and threshold will change] 
#Each feature corresponds to a weak learner, and is also called a tree-stump. 
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

class Images():
 	"""docstring for ClassName"""
 	numfaces = 11839
 	numnonfaces = 45357
 	def __init__(self, path, isface, samplesize, imgwidth=16):
 		#face16_000001.bmp
 		self.path = path
 		self.isface = isface
 		self.samplesize = samplesize
 		np.random.seed(2014)
		self.samples = range(1, self.numfaces if isface else self.numnonfaces)
 		np.random.shuffle(self.samples)
 		self.samples = self.samples[0:samplesize]
 		self.imgwidth = imgwidth
 	def LoadImages(self):
 		subpath = ("face16_" if self.isface else "nonface16_")
 		#+"{:0>6d}".format()
 		self.imgs = np.empty([self.samplesize,self.imgwidth, self.imgwidth])
 		for i in range(self.samplesize):
 			self.imgs[i] = misc.imread(self.path+subpath+"{:0>6d}".format(self.samples[i])+".bmp",flatten=True)
 			print "loading ","face.." if self.isface else "nonface..", self.samples[i]
 		return self.imgs

class Features():
	def __init__(self, imgwidth=16):
		self.imgwidth = 16
	def test_feature(self,img):
		assert(len(img.shape)==2)
		assert(img.shape[0]==img.shape[1]==self.imgwidth)
		sum1 = sum([img[i,j] for i in range(self.imgwidth) for j in range(self.imgwidth/2)])
		sum2 = sum([img[i,j] for i in range(self.imgwidth) for j in range(self.imgwidth/2, self.imgwidth)])
		return sum1-sum2



def test():
	psize = nsize = 10000
	f = Images("../newface16/",True, psize)
	nf = Images("../nonface16/",False, nsize)
	faces = f.LoadImages()
	nonfaces = nf.LoadImages()
	#plt.imshow(imgs[0], cmap=plt.cm.gray)
	#plt.show()
	pval = []
	nval = []
	feature = Features(16)
	for i in range(psize):
		pval.append(feature.test_feature(faces[i]))
		nval.append(feature.test_feature(nonfaces[i]))
	print len(pval), len(nval)
	plt.hist(pval, color="#6495ED", alpha=.5, bins=20, range=(-10000,10000))
	plt.hist(nval, color="#F08080", alpha=.5, bins=20, range=(-10000,10000))#, range=(-10000, +10000)
	plt.grid()
	plt.show()
	return 1

if __name__ == "__main__":
	test()