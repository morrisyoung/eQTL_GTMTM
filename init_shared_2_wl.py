import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import linear_model






## output: beta_cellfactor1 (from genetics to factors)

## for each factor, we use cis- SNPs of expressed genes in this factor as the input candidate SNPs, and we use LASSO to further sparsify








if __name__ == "__main__":




	##==== load data
	##
	X_raw = np.load("../../GTEx_data/data_genotype/X.npy")
	list_individual_raw = np.load("../../GTEx_data/data_genotype/list_individual.npy")
	repo = {}
	for i in range(len(list_individual_raw)):
		individual = list_individual_raw[i]
		data = X_raw[i]
		repo[individual] = data
	list_individual = np.load("../../GTEx_gtmnn/preprocess/data_prepared/Individual_list.npy")
	X = []
	for individual in list_individual:
		X.append(repo[individual])
	X = np.array(X)
	print "X shape:", X.shape

	F = np.load("./data_temp/F.npy")
	print "F shape:", F



	#
	N = len(X)
	I = len(X[0])
	D = len(F[0])
	print "shape:"
	print "N:", N
	print "I:", I
	print "D:", D





	##
	## for each factor: pick up candidate SNPs, do the LASSO
	##
	####========================================================================
	m_indi_snp = np.load("./data_temp/m_indi_snp.npy")
	beta = []
	for d in range(D):
		Data = np.multiply(X, m_indi_snp[d])			# X: (n_samples, n_features)
		Target = F[:, d] 								# Y: (n_samples, n_tasks)

		##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		#alpha = 0.0001
		#alpha = 0.0005
		alpha = 0.001
		#alpha = 0.005
		#alpha = 0.01
		##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		clf = linear_model.Lasso(alpha=alpha)
		clf.fit(Data, Target)

		#clf.coef_										# (n_features,)
		#clf.intercept_									# value
		temp = (clf.coef_).tolist()
		temp.append(clf.intercept_)

		beta.append(temp)
	init_beta = np.array(beta)							# (D, S+1)
	print "init_beta shape:",
	print init_beta.shape

	## test non-0 elements
	print "check num of non-0 snps each factor has (# of candidates, # of learned ones):"
	for d in range(D):
		print d, np.sum(m_indi_snp[d]), np.sum(np.sign(np.square(init_beta[d])))
	####========================================================================





	##==== save the init
	init_beta = init_beta.T 							# (S+1, D)
	print "final beta shape:", init_beta.shape
	np.save("./data_real_init/beta_cellfactor1", init_beta)





















