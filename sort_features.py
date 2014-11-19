import h5py
import numpy as np
import timeit
from multiprocessing import Process
import threading
import sys
savepath = "/Users/Shawn/cs276/"

numfaces = 11838
numnonfaces = 45356
unilabel = np.concatenate((np.ones(numfaces,dtype=bool), np.zeros(numnonfaces, dtype=bool)))

lock = threading.Lock()
def thread_sort(X, dataset, labels, slist, tid):
	count = 0
	for col in slist:
		count +=1
		tmp = np.concatenate((X[:,col].reshape(X.shape[0],1), unilabel.reshape(X.shape[0],1)), axis=1)
		temp = tmp[tmp[:,0].argsort(kind='heapsort')]
		lock.acquire()
		print "thread id", tid, 'sorted, might get stuck in storing'
		lock.release()
		dataset[:,col] = temp[:,0]
		labels[:,col] = temp[:,1]
		lock.acquire()
		print "thread id", tid, 'finished', count, 'out of', len(slist)
		lock.release()


def process_sort(i):
	X = np.load(savepath+'scores_feature_type'+str(i)+'.npy', mmap_mode='r')
	f = h5py.File(savepath+'sorted/scores_sorted_type'+str(i)+'.hdf5','w-')
	dataset = f.create_dataset("type"+str(i), X.shape, dtype=int)
	labels = f.create_dataset("labels", X.shape, dtype=bool)
	assert(X.shape[1]==2000)
	# sort each column
	threads = []
	for tid in range(500):
		slist = [4*tid+i for i in range(4)]
		t = threading.Thread(target=thread_sort, args=(X,dataset,labels,slist,tid,))
		t.start()
		threads.append(t)
		# Wait for all threads to complete
	for t in threads:
		t.join()
	f.close()
	print 'type'+str(i)+' sorted and converted!'


def sortAndConvert():
	start = timeit.default_timer()
	pros = []
	for i in range(1,7):
		p = Process(target=process_sort, args=(i,))
		p.start()
		pros.append(p)
	# Wait for all processes to complete
	for p in pros:
	    p.join()
	stop = timeit.default_timer()
	print "Time Used,", round(stop - start, 4)


if __name__ == "__main__":
	sortAndConvert()

