import numpy as np
import math
import timeit
import sys
from sklearn import linear_model






## rm the cis- effects of ALL the samples

## this is used after "init_shared_0_extractSamples.py"
## followed by individual scripts for linear model and tensor model

## dir: "~/GTEx_gtmtm/preprocess/"

## use total mean to impute before group-LASSO (group-LASSO needs full tensor to present), which is just 0 due to the global Z score

## NOTE: we need to reshape the X whenever we load it, since we don't have pre-splitted train/test sets any more






##==== to be filled later on
I = 0						# num of SNPs
J = 0						# num of genes
K = 0						# num of tissues
N = 0						# num of individuals




##==== variables
## NOTE: here we assume one chromosome model
##
X = []
Y = []						# a tensor



## NOTE: the following have the intercept term
init_beta_cis = []









if __name__ == "__main__":




	##========================================================================
	## loading data
	##========================================================================
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

	##
	Y = np.load("./data_raw/Data_final.npy")
	print "Y shape:", Y.shape


	##==== fill dimension
	I = len(X[0])
	J = len(Y[0][0])
	K = len(Y)
	N = len(Y[0])






	##============
	## train
	##============
	##========================================================================
	## init the Beta container
	mapping_cis = np.load("../../GTEx_gtmnn/preprocess/data_train/mapping_cis.npy")
	init_beta_cis = []
	for k in range(K):
		init_beta_cis.append([])
		for j in range(J):
			#temp = np.zeros(beta_cis[k][j].shape)
			amount = mapping_cis[j][1] - mapping_cis[j][0] + 1 + 1			## NOTE: the intercept
			temp = np.zeros(amount)
			init_beta_cis[k].append(temp)
		init_beta_cis[k] = np.array(init_beta_cis[k])
	init_beta_cis = np.array(init_beta_cis)




	####========================================================================
	## fill in the incomp tensor (with tissue mean) --> for group LASSO
	## here the global mean is just 0, as we take Z score for all samples of multi-tissues
	list_effective_pair = []
	for k in range(K):
		for n in range(N):
			if math.isnan(Y[k][n][0]):
				Y[k][n] = np.zeros(J)
			else:
				list_effective_pair.append((k, n))

	repo_k = {}
	repo_n = {}
	for pair in list_effective_pair:
		repo_k[pair[0]] = 1
		repo_n[pair[1]] = 1
	print "test coverage of tissues and individuals:", len(repo_k), len(repo_n)






	##============
	Y_tensor = Y
	for j in range(J):
		start = mapping_cis[j][0]
		end = mapping_cis[j][1]

		## for non-cis genes
		if (end - start + 1) == 0:
			for k in range(K):
				init_beta_cis[k][j] = np.array([np.average(Y_tensor[k, :, j])])
			continue

		##==== solve the group LASSO
		Data = X[:, start:end+1]									# X: (n_samples, n_features)
		Target = Y_tensor[:, :, j].T 								# Y: (n_samples, n_tasks)

		##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		## alpha 0.03 makes the program too slow and pick too many elements
		#alpha = 0.05												# TODO: to adapt this
		#alpha = 0.08												# TODO: to adapt this
		#alpha = 0.1												# TODO: to adapt this
		alpha = 0.15												# TODO: to adapt this
		#alpha = 0.2												# TODO: to adapt this
		## alpha 0.25 already makes the program pick only one SNP
		##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		clf = linear_model.MultiTaskLasso(alpha=alpha)				## NOTE: seems 0.1 is a good number
		clf.fit(Data, Target)

		#clf.coef_													# (n_tasks, n_features)
		#clf.intercept_												# (n_tasks,)
		intercept = (np.array([clf.intercept_])).T
		beta = np.concatenate((clf.coef_, intercept), axis=1)		# (K, (D+1))

		## save into init_beta_cis
		for k in range(K):
			init_beta_cis[k][j] = beta[k]
			#print np.sum(np.sign(np.square(beta[k])))

		## TEST
		#arr1 = np.sign(np.square(beta[0]))
		#print arr1.shape
		#print "arr1:"
		#for i in range(len(arr1)):
		#	if arr1[i] == 1:
		#		print i


	##============
	init_beta_cis = np.array(init_beta_cis)
	print "init_beta_cis shape:",
	print init_beta_cis.shape
	print "and data types of three levels:",
	print type(init_beta_cis),
	print type(init_beta_cis[0]),
	print type(init_beta_cis[0][0])
	np.save("./data_real_init/init_beta_cis", init_beta_cis)










	##=============================
	## cis- effects
	##=============================
	beta_cis = np.load("./data_real_init/init_beta_cis.npy")

	##=============================
	Y_cis_all = []
	for k in range(K):
		Y_cis = []
		for j in range(J):
			start = mapping_cis[j][0]
			end = mapping_cis[j][1]

			## for non-cis- and cis- genes
			if (end - start + 1) == 0:
				temp = np.zeros(N) + beta_cis[k][j][0]
				Y_cis.append(temp)
			else:
				X_sub = X[:, start:end+1]
				m_ones = np.ones((N, 1))
				X_sub = np.concatenate((X_sub, m_ones), axis=1)						# N x (amount+1)
				beta_sub = beta_cis[k][j]												# 1 x (amount+1)
				Y_sub = np.dot(X_sub, beta_sub)											# 1 x N
				Y_cis.append(Y_sub)
		Y_cis = np.array(Y_cis)
		Y_cis = Y_cis.T
		Y_cis_all.append(Y_cis)
	Y_cis_all = np.array(Y_cis_all)
	print "Y_cis_all shape:", Y_cis_all
	np.save("./data_raw/Y_cis_all", Y_cis_all)















