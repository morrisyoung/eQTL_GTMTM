import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn import linear_model







threshold = 335
init_beta1 = []
init_beta2 = []









if __name__ == "__main__":




	#
	X = np.load("./data_simu/X.npy")
	Y = np.load("./data_simu/Y.npy")
	print X.shape
	print Y.shape


	#
	X = X[:threshold]
	Y = Y[:threshold]
	print X.shape
	print Y.shape




	# init
	##==== fill dimension
	I = len(X[0])
	J = len(Y[0])
	N = len(X)
	D = 100										## TODO: manually set this
	print "shape:"
	print "I:", I
	print "J:", J
	print "N:", N
	print "D:", D


	## X append one
	array_ones = (np.array([np.ones(N)])).T
	X = np.concatenate((X, array_ones), axis=1)									# N x (I+1)










	##===============================================================================================
	## init#1: PCA + straight linear solver
	##===============================================================================================
	##
	## init_beta2
	##
	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = D
	pca = PCA(n_components=n_factor)
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


	##
	## init_beta1
	##
	##==== linear system
	# the linear system: X x beta = m_factor_before
	init_beta1 = np.linalg.lstsq(X, Y1)[0].T
	print "init_beta1 shape:",
	print init_beta1.shape







	## should not use LASSO, since the simulator is not sparse one
	"""
	##===============================================================================================
	## init#2: PCA + sparse linear solver (LASSO)
	##===============================================================================================
	##
	## init_beta2
	##
	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = D
	pca = PCA(n_components=n_factor)
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



	##
	## init_beta1
	##
	##==== linear system
	# the linear system between: X x Y1
	X = X[:, :-1]
	print X.shape

	Data = X									# X: (n_samples, n_features)
	Target = Y1 								# Y: (n_samples, n_tasks)
	clf = linear_model.Lasso(alpha=0.0001)
	clf.fit(Data, Target)
	#clf.coef_									# (n_tasks, n_features)
	#clf.intercept_								# (n_tasks,)
	intercept = (np.array([clf.intercept_])).T
	init_beta1 = np.concatenate((clf.coef_, intercept), axis=1)
	print "init_beta1 shape:",
	print init_beta1.shape
	"""







	##=====================================================================================================================
	##==== save the init
	##=====================================================================================================================
	np.save("./data_simu/beta1_init", init_beta1)
	np.save("./data_simu/beta2_init", init_beta2)















