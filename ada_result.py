import numpy as np

if __name__ == "__main__":
	# load the results
	alphas = np.loadtxt('results/alphas.csv', delimiter=',')
	features = np.loadtxt('results/selected_features.csv',  delimiter=',')
	weights = np.loadtxt('results/weights.csv',  delimiter=',')

	print alphas.shape
	print features.shape
	print weights