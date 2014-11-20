#Design a few types of features as we discussed in class.
# For each feature plot the histograms for the positive and negative populations. Determine the threshold.
#[note that as the samples change their weight over time, the histogram and threshold will change] 
#Each feature corresponds to a weak learner, and is also called a tree-stump. 
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import random
import sys

def GetTable(img):
	# fill the summed area table
	table = np.empty(img.shape, dtype=int)
	for m in range(img.shape[0]):
		for n in range(img.shape[1]):
			if m==0 and n==0:
				table[0,0] = img[0,0]
			elif m==0:
				table[m,n] = table[m,n-1] + img[m,n]
			elif n==0:
				table[m,n] = table[m-1,n] + img[m,n]
			else:
				table[m,n] = table[m,n-1] + table[m-1,n] - table[m-1,n-1] + img[m,n]
	return table

class Images():
 	def __init__(self, path, isface, samplesize, imgwidth=16):
 		#face16_000001.bmp
 		self.numfaces = 11838
 		self.numnonfaces = 45356
 		self.path = path
 		self.isface = isface
 		self.samplesize = samplesize
 		np.random.seed(2014)
		self.samples = range(1, self.numfaces+1 if isface else self.numnonfaces+1)
		flag = (samplesize==self.numfaces) if isface else (samplesize==self.numnonfaces)
		if not flag:
 			np.random.shuffle(self.samples)
 		self.samples = self.samples[0:samplesize]
 		self.imgwidth = imgwidth
 	
 	#### Load Image
 		subpath = ("face" if self.isface else "nonface") + str(self.imgwidth)+"_"
 		#+"{:0>6d}".format()
 		self.imgs = np.empty([self.samplesize,self.imgwidth, self.imgwidth], dtype=int)
 		self.tables = np.empty([self.samplesize,self.imgwidth, self.imgwidth], dtype=int)
 		for i in range(self.samplesize):
 			self.imgs[i] = misc.imread(self.path+subpath+"{:0>6d}".format(self.samples[i])+".bmp",flatten=True)
 			self.tables[i] = GetTable(self.imgs[i])
 		print ("Face" if isface else "Nonface") + " Loaded Successfully."
 		print "Number Loaded, ", samplesize
 		print "Imgwidth,", self.imgwidth
 	def GetImgs(self):
 		return self.imgs
 	def GetTables(self):
 		return self.tables


# generate random features
class Features():
	def __init__(self, imgwidth=16):
		self.imgwidth = imgwidth
	def setTable(self, table):
		self.table = table
	def rand_feature1(self,n=2000, vertical= True):
		# +-----------+
		# |XXXXXXXXXXX|
		# |XXXXXXXXXXX|
		# |           |
		# |           |
		# +-----------+


		#+-------------+
		#|XXXXXXX      |
		#|XXXXXXX      |
		#+-------------+

		# return: n*5 array of [x, y, row, col, midpos]
		print "\nGenerating Features: type 1"
		print "Vertical: ", vertical
		print "Number to generate:", n
		myset = set()
		#random.seed(2014)
		d1 = self.imgwidth
		d2 = pow(self.imgwidth,2)
		d3 = pow(self.imgwidth,3)
		d4 = pow(self.imgwidth,4)
		result = np.empty([n, 5],dtype=int)
		i = 0
		minrows = 2
		mincols = 2 
		while i<n:
			#min rownum 2
			x = random.randrange(self.imgwidth-minrows+1)
			y = random.randrange(self.imgwidth-mincols+1)
			rownum = random.randrange(minrows, self.imgwidth+1-x)
			colnum = random.randrange(mincols, self.imgwidth+1-y)
			midpos = random.randrange(x+1, x+rownum)
			curkey = x+y*d1+rownum*d2+colnum*d3+midpos*d4
			if curkey in myset:
				continue
			else:
				if vertical:
					result[i] = np.array([x, y, rownum, colnum, midpos])
				else:
					result[i] = np.array([y, x, colnum, rownum, midpos])
				i += 1
				myset.add(curkey)
		print "Complete! Result shape:", result.shape
		return result

	def rand_feature2(self, n=2000, vertical=True):
		# +----------+
		# |          |
		# |XXXXXXXXXX|
		# |XXXXXXXXXX|
		# |          |
		# +----------+


		# +-----------------+
		# |     XXXXXXX     |
		# |     XXXXXXX     |
		# |     XXXXXXX     |
		# +-----------------+

		# return: n*6 array of [x, y, row, col, midpos1, midpos2]
		# midpos1 is the start of the mid
		# midpos2 is the end (excluded) of the mid
		print "\nGenerating Features: type 2"
		print "Vertical: ", vertical
		print "Number to generate:", n
		myset = set()
		#random.seed(2014)
		d1 = self.imgwidth
		d2 = pow(self.imgwidth,2)
		d3 = pow(self.imgwidth,3)
		d4 = pow(self.imgwidth,4)
		d5 = pow(self.imgwidth,5)
		result = np.empty([n, 6], dtype=int)
		i = 0
		minrows = 3
		mincols = 2 
		while i<n:
			#min rownum 2
			x = random.randrange(self.imgwidth-minrows+1)
			y = random.randrange(self.imgwidth-mincols+1)
			rownum = random.randrange(minrows, self.imgwidth+1-x)
			colnum = random.randrange(mincols, self.imgwidth+1-y)
			midpos1 = random.randrange(x+1, x+rownum-1)
			midpos2 = random.randrange(midpos1+1, x+rownum)
			curkey = x+y*d1+rownum*d2+colnum*d3+midpos1*d4+midpos2*d5
			if curkey in myset:
				continue
			else:
				if vertical:
					result[i] = np.array([x, y, rownum, colnum, midpos1, midpos2])
				else:
					result[i] = np.array([y, x, colnum, rownum, midpos1, midpos2])
				i += 1
				myset.add(curkey)
		print "Complete! Result shape:", result.shape
		return result


	def rand_feature3(self, n=2000, vertical=True):
		# +--------------+
		# |XXXXXXX       |
		# |XXXXXXX       |
		# |XXXXXXX       |
		# |XXXXXXX       |
		# |       XXXXXXX|
		# |       XXXXXXX|
		# |       XXXXXXX|
		# +--------------+


		# return: n*6 array of [x, y, row, col, midpos1, midpos2]
		# midpos1 is the end of the first mid/start of the second part
		# midpos1 is the end of the first mid/start of the second part
		print "\nGenerating Features: type 3"
		print "Number to generate:", n
		myset = set()
		#random.seed(2014)
		d1 = self.imgwidth
		d2 = pow(self.imgwidth,2)
		d3 = pow(self.imgwidth,3)
		d4 = pow(self.imgwidth,4)
		d5 = pow(self.imgwidth,5)
		result = np.empty([n, 6], dtype=int)
		i = 0
		minrows = 2
		mincols = 2 
		while i<n:
			#min rownum 2
			x = random.randrange(self.imgwidth-minrows+1)
			y = random.randrange(self.imgwidth-mincols+1)
			rownum = random.randrange(minrows, self.imgwidth+1-x)
			colnum = random.randrange(mincols, self.imgwidth+1-y)
			midpos1 = random.randrange(x+1, x+rownum)
			midpos2 = random.randrange(y+1, y+colnum)
			curkey = x+y*d1+rownum*d2+colnum*d3+midpos1*d4+midpos2*d5
			if curkey in myset:
				continue
			else:
				result[i] = np.array([x, y, rownum, colnum, midpos1, midpos2])
				i += 1
				myset.add(curkey)
		print "Complete! Result shape:", result.shape
		return result

	def rand_feature4(self, n=2000, vertical=True):
		# +-----------------+
		# |                 |
		# | XXXXXXXXXXXXXXX |
		# | XXXXXXXXXXXXXXX |
		# | XXXXXXXXXXXXXXX |
		# |                 |
		# +-----------------+


		# return: n*8 array of [x, y, row, col, xx, yy, irow, icol]
		print "\nGenerating Features: type 4"
		print "Number to generate:", n
		myset = set()
		#random.seed(2014)
		d1 = self.imgwidth
		d2 = pow(self.imgwidth,2)
		d3 = pow(self.imgwidth,3)
		d4 = pow(self.imgwidth,4)
		d5 = pow(self.imgwidth,5)
		d6 = pow(self.imgwidth,6)
		d7 = pow(self.imgwidth,7)
		result = np.empty([n, 8], dtype=int)
		i = 0
		minrows = 3
		mincols = 3 
		while i<n:
			x = random.randrange(self.imgwidth-minrows+1)
			y = random.randrange(self.imgwidth-mincols+1)
			rownum = random.randrange(minrows, self.imgwidth+1-x)
			colnum = random.randrange(mincols, self.imgwidth+1-y)
			xx = random.randrange(x+1, x+rownum-1)
			yy = random.randrange(y+1, y+colnum-1)
			# xx+xrow-1 < x+rownum-1
			try:
				xrow = random.randrange(1, x+rownum-xx)
				yrow = random.randrange(1, y+colnum-yy)
			except ValueError:
				print x,y,rownum,colnum,xx,yy
				print ValueError
				sys.exit()
			curkey = x+y*d1+rownum*d2+colnum*d3+xx*d4+yy*d5+xrow*d6+yrow*d7
			if curkey in myset:
				continue
			else:
				result[i] = np.array([x, y, rownum, colnum, xx, yy, xrow, yrow])
				i += 1
				myset.add(curkey)
		print "Complete! Result shape:", result.shape
		return result
	def GetFeatureImg(self, feature, ftype):
		if ftype<=2:
			type = 1
			vertical = True if ftype==1 else False
		elif ftype<=4:
			type = 2
			vertical = True if ftype==3 else False
		else:
			## 5, 6 --> 3,4
			type = ftype - 2
		print "current feature,", feature
		print "type,", type
		testimage = np.zeros([self.imgwidth, self.imgwidth])
		testimage.fill(0)
		if type==1:
			for x in range(feature[0], feature[0]+feature[2]):
				for y in range(feature[1], feature[1]+feature[3]):
					flag = (x<feature[4] if vertical else y<feature[4])
					testimage[x][y] = -2 if flag else 1
					#print (x,y), flag
		elif type==2:
			for x in range(feature[0], feature[0]+feature[2]):
				for y in range(feature[1], feature[1]+feature[3]):
					flag = (x>=feature[4] and x<feature[5]) if vertical else (y>=feature[4] and y<feature[5])
					testimage[x][y] = -2 if flag else 1
					#print (x,y), flag
		elif type==3:
			print "type3 feature"
			for x in range(feature[0], feature[0]+feature[2]):
				for y in range(feature[1], feature[1]+feature[3]):
					flag = (x<feature[4] and y<feature[5]) or (x>=feature[4] and y>=feature[5])
					testimage[x][y] = -2 if flag else 1
					print (x,y), flag

		elif type==4:
			print "type4 feature"
			for x in range(feature[0], feature[0]+feature[2]):
				for y in range(feature[1], feature[1]+feature[3]):
					flag = (x>=feature[4] and x<feature[4]+feature[6]) and (y>=feature[5] and y<feature[5]+feature[7])
					testimage[x][y] = -2 if flag else 1
					print (x,y), flag
		return testimage


# for a given image, for a given type of features, get all the scores
class Scores():
	def __init__(self, table, features=None, ftype=1):
		self.table = table
		self.imgwidth = table.shape[0]
		self.features = features
		if ftype<=2:
			self.type = 1
			self.vertical = True if ftype==1 else False
		elif ftype<=4:
			self.type = 2
			self.vertical = True if ftype==3 else False
		else:
			## 5, 6 --> 3,4
			self.type = ftype - 2
		#print "\nNew Table Initialized. Feature Type:", self.type, "Vertical:", self.vertical, "Number of features:", features.shape[0]
		#print  "Preparing to calculate the scores.."
	def SetOneFeature(self, feature):
		self.features = np.array([feature])
	def SetfType(self, ftype):
		if ftype<=2:
			self.type = 1
			self.vertical = True if ftype==1 else False
		elif ftype<=4:
			self.type = 2
			self.vertical = True if ftype==3 else False
		else:
			## 5, 6 --> 3,4
			self.type = ftype - 2

	def getScore(self, feature):
			# +-----------+
			# |           |
			# +-----0-----0
			# |     |XXXXXX
			# |     |XXXXXX
			# |     |XXXXXX
			# +-----0XXXXX0
		if self.type==1:
			#  1        2 
			#   XXXXXXXXX  
			#   XXXXXXXXX  
			#  6XXXXXXXX3
			#   |       |  
			#  5+-------4  

			# 1   6    5
			#  XXXX+---+ 
			#  XXXX    | 
			#  XXXX    | 
			# 2XXX3+---4 
			
			# 4-2-5+1 - 2*(3-6-2+1)
			try:
				c1 = self.table[feature[0]-1, feature[1]-1]
			except IndexError:
				c1 = 0
			try:
				c2 = self.table[feature[0]-1, feature[1]+feature[3]-1] if self.vertical else self.table[feature[0]+feature[2]-1, feature[1]-1]
			except IndexError:
				c2 = 0
			try:
				c5 = self.table[feature[0]+feature[2]-1, feature[1]-1] if self.vertical else self.table[feature[0]-1, feature[1]+feature[3]-1]
			except IndexError:
				c5 = 0
			try:
				c6 = self.table[feature[4]-1, feature[1]-1] if self.vertical else self.table[feature[0]-1, feature[4]-1]
			except IndexError:
				c6 = 0
			# the indexError way won't work! it's numpy...
			if self.vertical:
				if feature[0]==0:
					c1=c2=0
				if feature[1]==0:
					c1=c6=c5=0
			else:
				if feature[0]==0:
					c1=c6=c5=0
				if feature[1]==0:
					c1=c2=0

			c4 = self.table[feature[0]+feature[2]-1, feature[1]+feature[3]-1]
			c3 = self.table[feature[4]-1, feature[1]+feature[3]-1] if self.vertical else self.table[feature[0]+feature[2]-1, feature[4]-1]
			return (c4-c2-c5+c1)-2*(c3-c6-c2+c1)
		elif self.type==2:
			# 1     2
			#  +----+ 
			# 8|    3 
			#  XXXXXX
			# 7XXXXX4
			#  |    | 
			# 6+----5

			# 1    8     7    6
			#  +----XXXXXX----+
			#  |    XXXXXX    |
			# 2+---3XXXXX4----5

			# (5-6-2+1)-2*(4-7-3+8)
			try:
				c1 = self.table[feature[0]-1, feature[1]-1]
			except IndexError:
				c1 = 0
			try:
				c2 = self.table[feature[0]-1, feature[1]+feature[3]-1] if self.vertical else self.table[feature[0]+feature[2]-1, feature[1]-1]
			except IndexError:
				c2 = 0
			try:
				c6 = self.table[feature[0]+feature[2]-1, feature[1]-1] if self.vertical else self.table[feature[0]-1, feature[1]+feature[3]-1]
			except IndexError:
				c6 = 0
			try:
				c8 = self.table[feature[4]-1, feature[1]-1] if self.vertical else self.table[feature[0]-1, feature[4]-1]
			except IndexError:
				c8 = 0
			try:
				c7 = self.table[feature[5]-1, feature[1]-1] if self.vertical else self.table[feature[0]-1, feature[5]-1]
			except IndexError:
				c7 = 0
			# the indexError way won't work! it's numpy...
			if self.vertical:
				if feature[0]==0:
					c1=c2=0
				if feature[1]==0:
					c1=c8=c7=c6=0
			else:
				if feature[0]==0:
					c1=c8=c7=c6=0
				if feature[1]==0:
					c1=c2=0

			c5 = self.table[feature[0]+feature[2]-1, feature[1]+feature[3]-1]
			c4 = self.table[feature[5]-1, feature[1]+feature[3]-1] if self.vertical else self.table[feature[0]+feature[2]-1, feature[5]-1]
			c3 = self.table[feature[4]-1, feature[1]+feature[3]-1] if self.vertical else self.table[feature[0]+feature[2]-1, feature[4]-1]
			return (c5-c6-c2+c1)-2*(c4-c7-c3+c8)

		elif self.type==3:
			# 1    2    9
			#  XXXXX----+
			#  XXXXX    |
			# 4XXXX3    8
			#  |    *XXXX
			#  |    XXXXX
			# 5+---6XXXX7
			try:
				c1 = self.table[feature[0]-1, feature[1]-1]
			except IndexError:
				c1 = 0
			try:
				c2 = self.table[feature[0]-1, feature[5]-1]
			except IndexError:
				c2 = 0
			try:
				c9 = self.table[feature[0]-1, feature[1]+feature[3]-1]
			except IndexError:
				c9 = 0
			try:
				c4 = self.table[feature[4]-1, feature[1]-1]
			except IndexError:
				c4 = 0
			try:
				c5 = self.table[feature[0]+feature[2]-1, feature[1]-1]
			except IndexError:
				c5 = 0
			# the indexError way won't work! it's numpy...
			if feature[0]==0:
				c1=c2=c9=0
			if feature[1]==0:
				c1=c4=c5=0


			c3 = self.table[feature[4]-1, feature[5]-1]
			c6 = self.table[feature[0]+feature[2]-1, feature[5]-1]
			c7 = self.table[feature[0]+feature[2]-1, feature[1]+feature[3]-1]
			c8 = self.table[feature[4]-1, feature[1]+feature[3]-1]
			return (c7-c5-c9+c1)-2*(c7-c6-c8+c3)-2*(c3-c4-c2+c1)
		elif self.type==4:
			# 1           2
			#  +----------+
			#  | 5    6   |
			#  |  XXXXX   |
			#  | 7XXXX8   |
			#  |          |
			# 3+----------4
			try:
				c1 = self.table[feature[0]-1, feature[1]-1]
			except IndexError:
				c1 = 0
			try:
				c2 = self.table[feature[0]-1, feature[1]+feature[3]-1]
			except IndexError:
				c2 = 0
			try:
				c3 = self.table[feature[0]+feature[2]-1, feature[1]-1]
			except IndexError:
				c3 = 0
			# the indexError way won't work! it's numpy...
			if feature[0]==0:
				c1=c2=0
			if feature[1]==0:
				c1=c3=0
			c4 = self.table[feature[0]+feature[2]-1, feature[1]+feature[3]-1]
			c5 = self.table[feature[4]-1, feature[5]-1]
			c6 = self.table[feature[4]-1, feature[5]+feature[7]-1]
			c7 = self.table[feature[4]+feature[6]-1, feature[5]-1]
			c8 = self.table[feature[4]+feature[6]-1, feature[5]+feature[7]-1]
			return (c4-c3-c2+c1)-2*(c8-c7-c6+c5)
	
	def getScores(self):
		result = np.empty(self.features.shape[0], dtype=int)
		for i in range(self.features.shape[0]):
			result[i] = self.getScore(self.features[i])
			# if i>100 and i%100==0:
			# 	#print "completed 100.."
		#print "All Complete!!"
		return result

def test():
	## the code below test:
	#1. the feature generation, passed
	#2. the table generation, passed
	#3. the score calculation, passed
	vflag =True
	width = 16
	type = 1
	f = Features(width)
	#result = f.rand_feature4(1, vertical=vflag)
	featureimg = f.GetFeatureImg([11,0,5,16,12], 1)
	print "feature image,\n", featureimg
	table = GetTable(featureimg)
	imgplot = plt.imshow(featureimg, cmap=plt.cm.gray, vmax=1, vmin=-2)
	imgplot.set_interpolation('nearest')
	plt.show()

def test1():
	# the code below test
	# load images
	psize = nsize = 10000
	f = Images("../newface16/",True, psize)
	faces = f.GetImgs()
	ftables = f.GetTables()
	nf = Images("../nonface16/",False, nsize)
	nonfaces = nf.GetImgs()
	nftables = nf.GetTables()
	#plt.imshow(imgs[0], cmap=plt.cm.gray)
	#plt.show()
	pval = []
	nval = []
	feature = Features(16)
	for i in range(psize):
		pval.append(feature.test_feature(faces[i], ftables[i]))
		nval.append(feature.test_feature(nonfaces[i], nftables[i]))
	print len(pval), len(nval)
	plt.hist(pval, color="#6495ED", alpha=.5, bins=20, range=(-200,300))
	plt.hist(nval, color="#F08080", alpha=.5, bins=20, range=(-200,300))#, range=(-10000, +10000)
	plt.grid()
	plt.show()
	return 1

if __name__ == "__main__":
	test()