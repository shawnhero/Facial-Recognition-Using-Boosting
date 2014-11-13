import h5py
import numpy as np
import timeit
from multiprocessing import Process

savepath = "/Users/Shawn/cs276/"


def convert(i):
	X = np.load(savepath+'scores_feature_type'+str(i)+'.npy', mmap_mode='r')
	f = h5py.File(savepath+'scores_feature_type'+str(i)+'.hdf5','w-')
	dset = f.create_dataset("type"+str(i), data=X)
	f.close()
	print 'type'+str(i)+' converted!'

if __name__ == "__main__":
	start = timeit.default_timer()
	pros = []
	for i in range(1,7):
		p = Process(target=convert, args=(i,))
		p.start()
		pros.append(p)
	# Wait for all processes to complete
	for p in pros:
	    p.join()
	stop = timeit.default_timer()
	print "Time Used,", round(stop - start, 4)