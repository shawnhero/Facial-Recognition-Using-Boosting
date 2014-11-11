import adaboosting as ada
import weaklearners as weak
import numpy as np
import timeit


def Combo( numfaces, numnonfaces, numfeatures, imgwidth=16, save=False):
	numfeatures = (numfeatures/6)*6
	faces = weak.Images("../newface"+str(imgwidth)+"/",isface=True, samplesize=numfaces)
	nonfaces = weak.Images("../nonface"+str(imgwidth)+"/",isface=False, samplesize=numnonfaces)

	# get the sum table for the images
	face_tables = faces.GetTables()
	nonface_tables = nonfaces.GetTables()
	train_faces_scores = np.empty([numfaces, numfeatures], dtype=int)
	train_nonfaces_scores = np.empty([numnonfaces, numfeatures], dtype=int)
	
	#Generate the features
	features = weak.Features(imgwidth)
	numsamples_per_feature = numfeatures/6
	f1v = features.rand_feature1(n=numsamples_per_feature, vertical=True)
	f1vf = features.rand_feature1(n=numsamples_per_feature, vertical=False)
	f2v = features.rand_feature2(n=numsamples_per_feature, vertical=True)
	f2vf = features.rand_feature2(n=numsamples_per_feature,vertical=False)
	f3 = features.rand_feature3(numsamples_per_feature)
	f4 = features.rand_feature4(numsamples_per_feature)

	## fill the score tables
	print "\nPreparing to fill the face score table.."
	print "Number of faces,", numfaces
	for i in range(numfaces):
		curscore = weak.Scores(face_tables[i], features=f1v, ftype=1, vertical=True)
		s1v = curscore.getScores()
		curscore = weak.Scores(face_tables[i], features=f1vf, ftype=1, vertical=True)
		s1vf = curscore.getScores()
		curscore = weak.Scores(face_tables[i], features=f2v, ftype=2, vertical=True)
		s2v = curscore.getScores()
		curscore = weak.Scores(face_tables[i], features=f2vf, ftype=2, vertical=False)
		s2vf = curscore.getScores()
		curscore = weak.Scores(face_tables[i], features=f3, ftype=3)
		s3 = curscore.getScores()
		curscore = weak.Scores(face_tables[i], features=f4, ftype=4)
		s4 = curscore.getScores()
		train_faces_scores[i]  = np.concatenate((s1v, s1vf, s2v, s2vf, s3, s4), axis=1)
		if i>=25 and i%25==0:
			print "completed", i
	print "All Complete! score table shape,", train_faces_scores.shape

	print "\nPreparing to fill the nonface score table.."
	print "Number of nonfaces,", numnonfaces
	for i in range(numnonfaces):
		curscore = weak.Scores(nonface_tables[i], features=f1v, ftype=1, vertical=True)
		s1v = curscore.getScores()
		curscore = weak.Scores(nonface_tables[i], features=f1vf, ftype=1, vertical=False)
		s1vf = curscore.getScores()
		curscore = weak.Scores(nonface_tables[i], features=f2v, ftype=2, vertical=True)
		s2v = curscore.getScores()
		curscore = weak.Scores(nonface_tables[i], features=f2vf, ftype=2, vertical=False)
		s2vf = curscore.getScores()
		curscore = weak.Scores(nonface_tables[i], features=f3, ftype=3)
		s3 = curscore.getScores()
		curscore = weak.Scores(nonface_tables[i], features=f4, ftype=4)
		s4 = curscore.getScores()
		train_nonfaces_scores[i]  = np.concatenate((s1v, s1vf, s2v, s2vf, s3, s4), axis=1)
		if i>=25 and i%25==0:
			print "completed", i
	print "All Complete! score table shape,", train_nonfaces_scores.shape
	if save:
		print "Saving all the feature scores.."
		np.savetxt('face_scores.csv', train_faces_scores, delimiter=',',fmt='%d')
		np.savetxt('nonface_scores.csv', train_nonfaces_scores, delimiter=',',fmt='%d')

	# ada.adaboosting(np.concatenate((train_faces_scores, train_nonfaces_scores), axis=0),  np.concatenate((np.ones(numfaces), np.zeros(numnonfaces)), T=50)


if __name__ == "__main__":
	start = timeit.default_timer()
	Combo(1000,1000,1000, 16, save=True)
	stop = timeit.default_timer()
	print "Time Used,", round(stop - start, 4)