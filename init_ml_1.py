import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import linear_model










## do the train/test split first of all
## then use the train data with PCA and (group-)LASSO to get sparse beta2, and beta1 indicator

## init procedure:
##	spread the incomp tensor to matrix;
##	do PCA;
##	average tissue effects;
##	fill in incomp tensor with mean, then solve group LASSO for each gene








##==== scale of the input data
J = 0						# num of genes
K = 0						# num of tissues
D = 0
N = 0						# num of individuals







if __name__ == "__main__":



	##=====================================================================================================================
	##==== load data (simu, incomp tensor)
	##=====================================================================================================================
	##
	Y = np.load("./data_raw/Data_final.npy")

	##
	## NOTE: take the residuals
	Y_cis_all = np.load("./data_raw/Y_cis_all.npy")
	Y = Y - Y_cis_all
	##

	##==== fill dimension
	J = len(Y[0][0])
	K = len(Y)
	N = len(Y[0])
	D = 400										## TODO: manually set this
	print "shape:"
	print "J:", J
	print "K:", K
	print "N:", N
	print "D:", D


	## to make: Y_train, Y_cis
	# list_effective_pair = []
	# for k in range(K):
	# 	for n in range(N):
	# 		if not math.isnan(Y[k][n][0]):
	# 			list_effective_pair.append((k, n))
	# print "there are # of effective samples:", len(list_effective_pair)
	# np.random.shuffle(list_effective_pair)
	# threshold = int( len(list_effective_pair) * 0.75 )
	# list_effective_pair_train = list_effective_pair[:threshold]
	# list_effective_pair_test = list_effective_pair[threshold:]
	# couldn't pass the coverage test
	# so sample by individual, and check tissue coverage
	list_effective_pair_train = []
	list_effective_pair_test = []
	for n in range(N):
		list_effective_pair = []
		for k in range(K):
			if not math.isnan(Y[k][n][0]):
				list_effective_pair.append((k, n))
		np.random.shuffle(list_effective_pair)
		threshold = int( len(list_effective_pair) * 0.25 )					# to pick up testing set first
		list_effective_pair_test += list_effective_pair[:threshold]
		list_effective_pair_train += list_effective_pair[threshold:]
	list_effective_pair_train = np.array(list_effective_pair_train)
	list_effective_pair_test = np.array(list_effective_pair_test)
	print "train samples:", len(list_effective_pair_train)
	print "test samples:", len(list_effective_pair_test)







	# check coverage of training set
	repo_tissue = {}
	repo_individual = {}
	for pair in list_effective_pair_train:
		k, n = pair[0], pair[1]
		repo_tissue[k] = 1
		repo_individual[n] = 1
	print "coverage checks:",
	print "tissue coverage:", len(repo_tissue), "of", K
	print "individual coverage:", len(repo_individual), "of", N

	# make Y_train, Y_test
	Y_train = np.zeros((K, N, J)) + float("Nan")
	for pair in list_effective_pair_train:
		k, n = pair[0], pair[1]
		Y_train[k][n] = Y[k][n]
	Y_test = np.zeros((K, N, J)) + float("Nan")
	for pair in list_effective_pair_test:
		k, n = pair[0], pair[1]
		Y_test[k][n] = Y[k][n]

	# save
	np.save("./data_raw/Y_train", Y_train)
	np.save("./data_raw/Y_test", Y_test)








	##=====================================================================================================================
	##==== cell factor (PCA + LASSO)
	##=====================================================================================================================
	##
	## init_beta_cellfactor2
	##
	####=============================== Scheme ===============================
	##	1. do PCA on sample matrix
	##	2. averaging the (Individual x Factor) matrix in order to eliminate the tissue effects, thus only individual effects left
	##	3. use these individual effects to retrieve their SNP causality
	##	4. use these individual effects to separately associate tissue effects of these factors
	##==== sample matrix
	Y_train_matrix = []
	Y_train_matrix_pos = []
	for k in range(K):
		for n in range(N):
			if not math.isnan(Y_train[k][n][0]):
				Y_train_matrix.append(Y_train[k][n])
				Y_train_matrix_pos.append((k, n))
	Y_train_matrix = np.array(Y_train_matrix)
	Y_train_matrix_pos = np.array(Y_train_matrix_pos)
	print "sample matrix shape (train):", Y_train_matrix.shape
	print "sample matrix pos shape (train):", Y_train_matrix_pos.shape

	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = D
	pca = PCA(n_components=n_factor)
	pca.fit(Y_train_matrix)
	Y2 = (pca.components_).T 						# Gene x Factor
	Y1 = pca.transform(Y_train_matrix)				# Sample x Factor
	variance = pca.explained_variance_ratio_

	##==== individual factors
	m_factor = np.zeros((N, D))
	list_count = np.zeros(N)
	for i in range(len(Y1)):
		k, n = Y_train_matrix_pos[i]
		m_factor[n] += Y1[i]
		list_count[n] += 1
	for n in range(N):
		m_factor[n] = m_factor[n] / list_count[n]


	####========================================================================
	## fill in the incomp tensor (with tissue mean) --> for group LASSO
	## here the global mean is just 0, as we take Z score for all samples of multi-tissues
	for k in range(K):
		for n in range(N):
			if math.isnan(Y_train[k][n][0]):
				Y_train[k][n] = np.zeros(J)


	####========================================================================
	## solve the group LASSO
	beta_tensor = []
	for j in range(J):
		Data = m_factor						# X: (n_samples, n_features)
		Target = Y_train[:, :, j].T 		# Y: (n_samples, n_tasks)

		clf = linear_model.MultiTaskLasso(alpha=0.01)
		clf.fit(Data, Target)

		#clf.coef_							# (n_tasks, n_features)
		#clf.intercept_						# (n_tasks,)
		intercept = (np.array([clf.intercept_])).T
		beta = np.concatenate((clf.coef_, intercept), axis=1)
		beta_tensor.append(beta)
	beta_tensor = np.array(beta_tensor)
	# beta_tensor: (J, K, (D+1))
	init_beta_cellfactor2 = np.transpose(beta_tensor, (1, 2, 0))
	print "init_beta_cellfactor2 shape (exp: K, D+1, J):",
	print init_beta_cellfactor2.shape








	## checked above code (Mar.24)








	##=====================================================================================================================
	##==== saving
	##=====================================================================================================================
	## save the init
	np.save("./data_real_init/beta_cellfactor2", init_beta_cellfactor2)

	## save factors: m_factor
	np.save("./data_temp/F", m_factor)

	## save non-0 genes for each factor
	print "non-zero entries for each factor (all tissues):"
	m_indi = []
	for d in range(D):
		#m_factor = init_beta_cellfactor2[:, :, d]
		m_factor = init_beta_cellfactor2[:, d, :]
		m_factor = np.square(m_factor)
		a_factor = np.sum(m_factor, axis=0)
		indi_factor = np.sign(a_factor)
		m_indi.append(indi_factor)
	m_indi = np.array(m_indi)
	print m_indi.shape
	np.save("./data_temp/m_indi", m_indi)

	## check and save effective SNPs for each factor
	print "output num of active genes and num of corresponding snps in each factor:"
	X = np.load("../../GTEx_gtmnn/preprocess/data_train/X.npy")
	I = len(X[0])
	#mapping_cis = np.load("../preprocess/data_train/mapping_cis.npy")
	mapping_cis = np.load("../../GTEx_gtmnn/preprocess/data_train/mapping_cis.npy")
	m_indi_snp = []
	for d in range(D):
		indi_snp = np.zeros(I)

		indi_factor = m_indi[d]
		for j in range(J):
			if indi_factor[j] == 1:				# gene in this factor
				start = mapping_cis[j][0]
				end = mapping_cis[j][1]
				for index in range(start, end+1):
					indi_snp[index] = 1
		print d, np.sum(indi_factor), np.sum(indi_snp)
		m_indi_snp.append(indi_snp)
	m_indi_snp = np.array(m_indi_snp)
	print m_indi_snp.shape
	np.save("./data_temp/m_indi_snp", m_indi_snp)








	## checked, to run










