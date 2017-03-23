## 


## for the first layer, given the individual factors and the candidate cis- SNPs from the subset of genes for that factor (all tissues)
## we can solve each factor separately, so parallel all the jobs







import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import linear_model




##==== scale of the input data
I = 0						# num of SNPs
D = 0
N = 0						# num of individuals


##==== variables
## NOTE: here we assume one chromosome model
X = []						# matrix of Individuals x SNPs
F = []						# factors of Individuals x D

init_beta_cellfactor1 = []		# matrix of first layer cell factor beta






##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============







if __name__ == "__main__":







	##=====================================================================================================================
	##==== load data (simu, incomp tensor)
	##=====================================================================================================================
	#
	#X = np.load("./data_train/X.npy")
	X = np.load("../preprocess/data_train/X.npy")
	#
	F = np.load("./data_temp/F.npy")
	#F = np.load("../preprocess/data_temp/F.npy")

	##==== fill dimension
	I = len(X[0])
	N = len(X)
	D = len(F[0])
	print "shape:"
	print "I:", I
	print "N:", N
	print "D:", D

	#init_beta_cellfactor1 = np.zeros(beta_cellfactor1.shape)
	init_beta_cellfactor1 = np.zeros((D, I+1))







	##
	##
	## for each factor: pick up candidate SNPs, do the LASSO
	##
	##
	####========================================================================
	m_indi_snp = np.load("./data_temp/m_indi_snp.npy")
	beta = []
	for d in range(D):
		Data = np.multiply(X, m_indi_snp[d])			# X: (n_samples, n_features)
		Target = F[:, d] 								# Y: (n_samples, n_tasks)

		clf = linear_model.Lasso(alpha=0.01)
		clf.fit(Data, Target)

		#clf.coef_							# (n_features,)
		#clf.intercept_						# value
		temp = (clf.coef_).tolist()
		temp.append(clf.intercept_)

		beta.append(temp)
	init_beta_cellfactor1 = np.array(beta)
	print "init_beta_cellfactor1 shape:",
	print init_beta_cellfactor1.shape

	## test non-0 elements
	print "check num of non-0 snps each factor has:"
	for d in range(D):
		print d, np.sum(m_indi_snp[d]), np.sum(np.sign(np.square(init_beta_cellfactor1[d])))
	####========================================================================










	##=====================================================================================================================
	##==== save the init
	##=====================================================================================================================
	np.save("./data_real_init/beta_cellfactor1", init_beta_cellfactor1)


	print "done..."












	#### DEBUG
	'''
	####========================================================================
	m_indi_snp = np.load("../preprocess/data_temp/m_indi_snp.npy")
	d = 0
	Data = np.multiply(X, m_indi_snp[d])			# X: (n_samples, n_features)
	Target = F[:, d] 								# Y: (n_samples, n_tasks)
	clf = linear_model.Lasso(alpha=0.01)
	clf.fit(Data, Target)
	#clf.coef_							# (n_features,)
	#clf.intercept_						# value
	beta = (clf.coef_).tolist()
	beta.append(clf.intercept_)
	beta = np.array(beta)
	print beta.shape
	print d, np.sum(m_indi_snp[d]), np.sum(np.sign(np.square(beta)))
	####========================================================================
	'''


