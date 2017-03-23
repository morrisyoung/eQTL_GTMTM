import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn import linear_model








init_beta1 = []
init_beta2 = []









if __name__ == "__main__":




	##==========================
	##==== split training and testing samples
	##==========================
	# in this case we use all the samples (the imcompleteness then become missing ceils)
	Y = np.load("./data_real/Y.npy")
	print Y.shape


	# init
	##==== fill dimension
	K = len(Y)
	N = len(Y[0])
	J = len(Y[0][0])
	D = 100										## TODO: manually set this
	print "shape:"
	print "K:", K
	print "N:", N
	print "J:", J
	print "D:", D


	## need to do the following before init, since will only use train data to init model
	shape = Y.shape
	dimension1 = shape[0]
	dimension2 = shape[1]
	dimension3 = shape[2]
	amount_sample = dimension1 * dimension2 * dimension3
	list_index_sample = np.arange(amount_sample)
	np.random.shuffle(list_index_sample)
	threshold = int(amount_sample * 0.75)						# threshold: to tune
	list_index_sample_train = list_index_sample[:threshold]
	list_index_sample_test = list_index_sample[threshold:]
	np.save("./data_real/list_index_sample_train", list_index_sample_train)
	np.save("./data_real/list_index_sample_test", list_index_sample_test)

	print len(list_index_sample_train),
	print len(list_index_sample_test)













	##==========================
	##==== reformat and save the training set Sample x Gene matrix (with Nan in) under "./data_simu/"
	##==========================
	Data = np.zeros((K, N, J)) + float("Nan")
	print Data

	for index in list_index_sample_train:
		k = index / (N*J)
		left = index % (N*J)
		i = left / J
		j = left % J
		Data[k][i][j] = Y[k][i][j]
	np.save("./data_real/train_incomp_tensor", Data)
	print Data


	##==========================
	##==== fill in the incomplete tensor with mean gene exp level
	##==========================
	m_count = np.zeros((K, J))
	m_sum = np.zeros((K, J))
	for k in range(K):
		for i in range(N):
			for j in range(J):
				if not math.isnan(Data[k][i][j]):
					m_count[k][j] += 1
					m_sum[k][j] += Data[k][i][j]
	for k in range(K):
		for j in range(J):
			m_sum[k][j] = m_sum[k][j] / m_count[k][j]
	for k in range(K):
		for i in range(N):
			for j in range(J):
				if math.isnan(Data[k][i][j]):
					Data[k][i][j] = m_sum[k][j]
	print Data







	##===============================================================================================
	## PCA
	##===============================================================================================
	Y1 = np.zeros((N, D))
	for k in range(K):
		n_factor = D
		pca = PCA(n_components=n_factor)
		Y = Data[k]
		pca.fit(Y)
		Y1 += pca.transform(Y)						# Sample x Factor
	Y1 = Y1 / K
	## Y1 is from the training samples, and fixed as the individual factors
	np.save("./data_real/m_factor_init", Y1)















