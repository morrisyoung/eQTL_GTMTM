import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn import linear_model



## extract the training samples and testing samples
## fill in the incomplete training matrix with sample mean
## init model with PCA + LASSO solver








init_beta1 = []
init_beta2 = []








if __name__ == "__main__":






	##==========================
	##==== split training and testing samples
	##==========================
	# in this case we use all the samples (the imcompleteness then become missing ceils)
	Y = np.load("./data_simu/Y.npy")
	print Y.shape


	# init
	##==== fill dimension
	J = len(Y[0])
	N = len(Y)
	D = 100										## TODO: manually set this
	print "shape:"
	print "J:", J
	print "N:", N
	print "D:", D


	## need to do the following before init, since will only use train data to init model
	amount_sample = len(Y) * len(Y[0])
	list_index_sample = np.arange(amount_sample)
	np.random.shuffle(list_index_sample)
	threshold = int(amount_sample * 0.75)						# threshold: to tune
	list_index_sample_train = list_index_sample[:threshold]
	list_index_sample_test = list_index_sample[threshold:]
	np.save("./data_simu/list_index_sample_train", list_index_sample_train)
	np.save("./data_simu/list_index_sample_test", list_index_sample_test)

	print len(list_index_sample_train),
	print len(list_index_sample_test)








	##==========================
	##==== reformat and save the training set Sample x Gene matrix (with Nan in) under "./data_simu/"
	##==========================
	Data = np.zeros((N, J)) + float("Nan")
	print Data

	for index in list_index_sample_train:
		i = index / J
		j = index % J
		Data[i][j] = Y[i][j]
	np.save("./data_simu/train_incomp_matrix", Data)
	print Data








	##==========================
	##==== fill in the incomplete tensor with mean gene exp level
	##==========================
	list_count = np.zeros(J)
	list_sum = np.zeros(J)
	for i in range(len(Data)):
		for j in range(len(Data[i])):
			if not math.isnan(Data[i][j]):
				list_count[j] += 1
				list_sum[j] += Data[i][j]
	for j in range(J):
		list_sum[j] = list_sum[j] / list_count[j]
	for i in range(len(Data)):
		for j in range(len(Data[i])):
			if math.isnan(Data[i][j]):
				Data[i][j] = list_sum[j]
	print Data







	## should not use the LASSO solver, as the simulator is not a sparse one
	"""
	##===============================================================================================
	## init: PCA + sparse linear solver (LASSO)
	##===============================================================================================
	##
	## init_beta2
	##
	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = 100								# TODO
	pca = PCA(n_components=n_factor)
	Y = Data
	pca.fit(Y)
	Y2 = (pca.components_).T 					# Gene x Factor
	Y1 = pca.transform(Y)						# Sample x Factor
	variance = pca.explained_variance_ratio_
	print "variance:", variance

	# linear system between: Y1, Y
	Data = Y1									# X: (n_samples, n_features)
	Target = Y 									# Y: (n_samples, n_tasks)
	clf = linear_model.Lasso(alpha=0.001)
	clf.fit(Data, Target)
	#clf.coef_									# (n_tasks, n_features)
	#clf.intercept_								# (n_tasks,)
	intercept = (np.array([clf.intercept_])).T
	init_beta2 = np.concatenate((clf.coef_, intercept), axis=1)
	print "init_beta2 shape:",
	print init_beta2.shape


	## Y1 is from the training samples, and fixed as the individual factors


	##
	## init_beta1
	##
	##==== linear system
	# the linear system between: X x Y1
	# in this case we use all the samples (the imcompleteness then become missing ceils)
	X = np.load("./data_simu/X.npy")
	Data = X									# X: (n_samples, n_features)
	Target = Y1 								# Y: (n_samples, n_tasks)
	clf = linear_model.Lasso(alpha=0.001)
	clf.fit(Data, Target)
	#clf.coef_									# (n_tasks, n_features)
	#clf.intercept_								# (n_tasks,)
	intercept = (np.array([clf.intercept_])).T
	init_beta1 = np.concatenate((clf.coef_, intercept), axis=1)
	print "init_beta1 shape:",
	print init_beta1.shape
	"""






	##===============================================================================================
	## init#1: PCA + straight linear solver
	##===============================================================================================
	##
	## init_beta2
	##
	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = D
	pca = PCA(n_components=n_factor)
	Y = Data
	pca.fit(Y)
	Y2 = (pca.components_).T 					# Gene x Factor
	Y1 = pca.transform(Y)						# Sample x Factor
	variance = pca.explained_variance_ratio_
	print variance
	#
	intercept = np.random.normal(size=(J, 1))
	init_beta2 = np.concatenate((Y2, intercept), axis=1)
	print "init_beta2 shape:",
	print init_beta2.shape

	## Y1 is from the training samples, and fixed as the individual factors

	##
	## init_beta1
	##
	##==== linear system
	# the linear system: X x beta = m_factor_before
	X = np.load("./data_simu/X.npy")
	## X append one
	array_ones = (np.array([np.ones(len(X))])).T
	X = np.concatenate((X, array_ones), axis=1)									# N x (I+1)
	##
	init_beta1 = np.linalg.lstsq(X, Y1)[0].T
	print "init_beta1 shape:",
	print init_beta1.shape








	##=====================================================================================================================
	##==== save the init
	##=====================================================================================================================
	#np.save("./data_simu/beta1_init", init_beta1)
	np.save("./data_simu/m_factor_init", Y1)
	#np.save("./data_simu/beta2_init", init_beta2)















