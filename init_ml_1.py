## 

## procedure:
##	spread the incomp tensor to matrix;
##	do PCA;
##	average tissue effects;
##	fill in incomp tensor with tissue mean, then solve group LASSO for each gene
##	visual the distribution of genes for different factors (400)






## probably just use the avaliable samples to solve the LASSO gene factor matrix?










import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import linear_model




##==== scale of the input data
I = 0						# num of SNPs
J = 0						# num of genes
K = 0						# num of tissues
D = 0
N = 0						# num of individuals


##==== variables
## NOTE: here we assume one chromosome model
Y = []						# tensor of gene expression (incomplete)
Y_pos = []					# list of pos

init_beta_cellfactor2 = []		# tensor (tissue specific) of second layer cell factor beta






##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============






if __name__ == "__main__":






	##=====================================================================================================================
	##==== load data (simu, incomp tensor)
	##=====================================================================================================================
	#
	X = np.load("../preprocess/data_train/X.npy")
	# Y and Y_pos
	K = 28										## TODO: specify the number of tissues
	Y = []
	Y_pos = []
	for k in range(K):
		data = np.load("../preprocess/data_train/Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load("../preprocess/data_train/Tensor_tissue_" + str(k) + "_pos.npy")
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)



	##
	##
	##
	##
	## NOTE: take the residuals
	Y_cis_train = np.load("../workbench54/data_real_init/Y_cis_train.npy")
	Y = Y - Y_cis_train
	##
	##
	##
	##



	##==== fill dimension
	I = len(X[0])
	J = len(Y[0][0])
	K = len(Y)
	N = len(X)
	D = 400										## TODO: manually set this
	print "shape:"
	print "I:", I
	print "J:", J
	print "K:", K
	print "N:", N
	print "D:", D

	#init_beta_cellfactor2 = np.zeros(beta_cellfactor2.shape)
	init_beta_cellfactor2 = np.zeros((K, J, D+1))










	##=====================================================================================================================
	##==== cell factor
	##=====================================================================================================================
	Y_cellfactor = Y
	##
	## init_beta_cellfactor2
	##
	####=============================== Scheme ===============================
	##	1. do PCA on sample matrix
	##	2. averaging the (Individual x Factor) matrix in order to eliminate the tissue effects, thus only individual effects left
	##	3. use these individual effects to retrieve their SNP causality
	##	4. use these individual effects to separately associate tissue effects of these factors
	##==== sample matrix
	Y_matrix = []
	Y_matrix_pos = []
	for i in range(len(Y_cellfactor)):
		for j in range(len(Y_cellfactor[i])):
			Y_matrix.append(Y_cellfactor[i][j])
			Y_matrix_pos.append(Y_pos[i][j])
	Y_matrix = np.array(Y_matrix)
	print "sample matrix shape:", Y_matrix.shape

	##==== do PCA for Sample x Gene, with number of factors as D
	n_factor = D
	pca = PCA(n_components=n_factor)
	pca.fit(Y_matrix)
	Y2 = (pca.components_).T 					# Gene x Factor
	Y1 = pca.transform(Y_matrix)				# Sample x Factor
	variance = pca.explained_variance_ratio_

	##==== individual factors
	m_factor = np.zeros((N, D))
	list_count = np.zeros(N)
	for i in range(len(Y1)):
		pos = Y_matrix_pos[i]
		m_factor[pos] += Y1[i]
		list_count[pos] += 1
	for n in range(N):
		m_factor[n] = m_factor[n] / list_count[n]
	####========================================================================





	## tune factor matrix into [0.1, 0.9]
	value_max = np.amax(m_factor)
	value_min = np.amin(m_factor)
	m_factor_tune = (m_factor - value_min) * (1 / (value_max - value_min))
	m_factor_tune = 0.5 + 0.8 * (m_factor_tune - 0.5)






	## fill in the incomp tensor (with tissue mean)
	####========================================================================
	##==== sample matrix
	Y_matrix = []
	Y_matrix_pos = []
	for i in range(len(Y_cellfactor)):
		for j in range(len(Y_cellfactor[i])):
			Y_matrix.append(Y_cellfactor[i][j])
			Y_matrix_pos.append(Y_pos[i][j])
	Y_matrix = np.array(Y_matrix)
	print "sample matrix shape:", Y_matrix.shape

	#
	m_ave = np.zeros((N, J))
	list_count = np.zeros(N)
	for i in range(len(Y_matrix)):
		pos = Y_matrix_pos[i]
		m_ave[pos] += Y_matrix[i]
		list_count[pos] += 1
	for n in range(N):
		m_ave[n] = m_ave[n] / list_count[n]

	# fill in all with ave, then substitute avaliable samples
	Y_tensor = np.zeros((K, N, J))
	for n in range(N):
		a_ave = m_ave[n]
		for k in range(K):
			Y_tensor[k][n] = a_ave
	for k in range(K):
		for n in range(len(Y_pos[k])):
			array = Y[k][n]
			pos = Y_pos[k][n]
			Y_tensor[k][pos] = array
	####========================================================================










	## solve the group LASSO
	####========================================================================
	beta_tensor = []
	for j in range(J):
		Data = m_factor_tune				# X: (n_samples, n_features)
		Target = Y_tensor[:, :, j].T 		# Y: (n_samples, n_tasks)

		clf = linear_model.MultiTaskLasso(alpha=0.01)
		clf.fit(Data, Target)

		#clf.coef_							# (n_tasks, n_features)
		#clf.intercept_						# (n_tasks,)
		intercept = (np.array([clf.intercept_])).T
		beta = np.concatenate((clf.coef_, intercept), axis=1)
		beta_tensor.append(beta)
	# beta_tensor: (J, K, (D+1))
	for k in range(K):
		for j in range(J):
			for d in range(D+1):
				init_beta_cellfactor2[k][j][d] = beta_tensor[j][k][d]
	print "init_beta_cellfactor2 shape:",
	print init_beta_cellfactor2.shape
	####========================================================================


	## solve the LASSO for tissues differently (non- group LASSO)
	'''
	####========================================================================
	beta_tensor = []
	for j in range(J):
		Data = m_factor_tune				# X: (n_samples, n_features)
		Target = Y_tensor[:, :, j].T 		# Y: (n_samples, n_tasks)
		clf = linear_model.Lasso(alpha=0.001)
		clf.fit(Data, Target)
		#clf.coef_							# (n_tasks, n_features)
		#clf.intercept_						# (n_tasks,)
		intercept = (np.array([clf.intercept_])).T
		beta = np.concatenate((clf.coef_, intercept), axis=1)
		beta_tensor.append(beta)
	# beta_tensor: (J, K, (D+1))
	for k in range(K):
		for j in range(J):
			for d in range(D+1):
				init_beta_cellfactor2[k][j][d] = beta_tensor[j][k][d]
	print "init_beta_cellfactor2 shape:",
	print init_beta_cellfactor2.shape
	####========================================================================
	'''










	##=====================================================================================================================
	##==== saving
	##=====================================================================================================================
	## save the init
	np.save("./data_real_init/beta_cellfactor2", init_beta_cellfactor2)
	## save factors (before sigmoid function)
	m_factor_before = np.zeros(m_factor_tune.shape)
	for n in range(N):
		for d in range(D):
			x = m_factor_tune[n][d]
			m_factor_before[n][d] = np.log( x / (1-x) )
	np.save("./data_temp/F", m_factor_before)
	## save non-0 genes for each factor
	print "non-zero entries for each factor (all tissues):"
	m_indi = []
	for d in range(D):
		m_factor = init_beta_cellfactor2[:, :, d]
		m_factor = np.square(m_factor)
		a_factor = np.sum(m_factor, axis=0)
		indi_factor = np.sign(a_factor)
		m_indi.append(indi_factor)
	m_indi = np.array(m_indi)
	print m_indi.shape
	np.save("./data_temp/m_indi", m_indi)
	## check and save effective SNPs for each factor
	print "output num of active genes and num of corresponding snps in each factor:"
	mapping_cis = np.load("../preprocess/data_train/mapping_cis.npy")
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


	##
	print "done..."






