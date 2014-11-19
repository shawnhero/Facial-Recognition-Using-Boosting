def FindFeatureError(self,col):
	# understand the threshold
	# http://stackoverflow.com/questions/9777282/the-best-way-to-calculate-the-best-threshold-with-p-viola-m-jones-framework
	print "finding threshold.."
	# find the threshold of the col-th classifier
	# find positive and negative mean
	pmask = np.asarray(self.labels, dtype=bool)
	nmask = np.asarray(1-self.labels, dtype=bool)
	pmean = np.mean(self.scores[pmask, col])
	nmean = np.mean(self.scores[nmask, col])
	# find the threshold between the two means
	step = np.fabs(pmean-nmean)/100
	thresholdlist = [(pmean if pmean<nmean else nmean)+step*i for i in range(101)]
	
	minerror = 1
	decisions = None
	errors = []
	for t in thresholdlist:
		flag_below = True if pmean<nmean else False
		cur_decisions = np.asarray([flag_below if self.scores[j,col]<=t else 1- flag_below for j in range(self.num_sample)], dtype=bool)
		error = matrix(np.fabs(cur_decisions - self.labels).reshape(1, self.num_sample))*matrix(self.weights.reshape(self.num_sample,1))
		error = error[0,0]
		if error < minerror:
			minerror = error
			decisions = cur_decisions
		## no use
		errors.append(error)
		print 'one threshold test passed out of 100'
	
	if 1-max(errors) < min(errors):
		print "Need a flip over! Didn't expect that could happen.."

	# threshold = thresholdlist[errors.index(min(errors))]
	# print "pmean", pmean
	# print "nmean", nmean
	# print "threshold:", threshold

	# calculate the decision of that feature
	print 'Decision found for','type'+str(self.ftype), 'col'+str(col)
	
	return minerror, decisions