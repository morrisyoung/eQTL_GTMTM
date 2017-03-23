## do the mini-batch gradient descent

## NOTE:
##	1. dimension indicators should be used whenever needed, rather than the len(Var) (as input will be appended to the intercept term)
##	2. batch has consistent effects across different tissues (so we don't have tissue-specific parameters)

## NOTE:
##	1. in this script, I use (n x k) to index the data, so I need to reshape beta everytime (from (d x k) to (k x d)); data should normally have (k x n) shape






import numpy as np
import math
import timeit
import sys






##==== learning setting
## TODO: to determine some parameters
num_iter = 100							# for simu data
#num_iter = 500							# for real data

#rate_learn = 0.0001					# for brain and chr22
#rate_learn = 0.00001					# for 10% of real scale
rate_learn = 0.0000001					# for 10% of real scale, init (as para from init is too weird)
#rate_learn = 0.0000001					# for real scale data







##==== to be filled later on
I = 0						# num of SNPs
J = 0						# num of genes
K = 0						# num of tissues
N = 0						# num of individuals
D = 0						# num of cell factors






##==== variables
## NOTE: here we assume one chromosome model
##
X = []						# matrix of Individuals x SNPs
Y = []
Y_pos = []
##
X_test = []						# matrix of Individuals x SNPs
Y_test = []
Y_pos_test = []



## NOTE: the following have the intercept term
beta_cellfactor1 = []		# matrix of first layer cell factor beta
beta_cellfactor2 = []		# tensor (tissue specific) of second layer cell factor beta
# the following corresponds to the above
der_cellfactor1 = []
der_cellfactor2 = []


#
list_sample = []




## cis- pre-calculated components
Y_cis_train = []
repo_Y_cis_train = {}
Y_cis_test = []







##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============







##==== forward/backward propogation, and gradient descent
def forward_backward_gd():
	global X, Y, Y_pos
	global beta_cellfactor1, beta_cellfactor2
	global der_cellfactor1, der_cellfactor2
	global I, J, K, D, N

	global rate_learn

	global list_sample

	global Y_cis_train
	global Y_cis_test



	print "forward_backward_gd..."



	##==========================================================================================
	## refill der containers, all with 0's (to make the GD more general)
	##==========================================================================================
	der_cellfactor1 = np.zeros(beta_cellfactor1.shape)
	der_cellfactor2 = np.zeros(beta_cellfactor2.shape)





	##==========================================================================================
	## prep for the mini-batch (across all tissues)
	##==========================================================================================
	## pick up some samples, and re-organize them into different tissues
	#
	size_batch = 100							## TODO: specify the batch size (across all tissues)
	list_pos = np.random.permutation(len(list_sample))[:size_batch]
	list_sample_batch = list_sample[list_pos]

	# re-organize the samples in this mini-batch into tissues
	Y_batch = []
	Y_pos_batch = []
	list_tissue_batch = []
	rep_tissue = {}
	for sample in list_sample_batch:
		pair = sample.split('-')
		tissue = int(pair[0])
		pos_sample = int(pair[1])
		sample_expr = Y[tissue][pos_sample]
		pos_individual = Y_pos[tissue][pos_sample]

		if tissue in rep_tissue:
			Y_batch[rep_tissue[tissue]].append(sample_expr)
			Y_pos_batch[rep_tissue[tissue]].append(pos_individual)
		else:
			Y_batch.append([sample_expr])
			Y_pos_batch.append([pos_individual])

			rep_tissue[tissue] = len(Y_batch) - 1			# index in the new incomp tensor
			list_tissue_batch.append(tissue)

	for i in range(len(Y_batch)):
		Y_batch[i] = np.array(Y_batch[i])
		Y_pos_batch[i] = np.array(Y_pos_batch[i])
	Y_batch = np.array(Y_batch)
	Y_pos_batch = np.array(Y_pos_batch)
	list_tissue_batch = np.array(list_tissue_batch)



	## several key components to be used later on:
	#Y_batch = []
	#Y_pos_batch = []
	#list_tissue_batch = []
	## please avoid re-naming




	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##==========================================================================================
	## forward prop
	##==========================================================================================
	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##=============
	## from cell factor (all tissues)
	##=============
	# first layer
	beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
	m_factor_before = np.dot(X, beta_cellfactor1_reshape)					# size_batch x D

	# logistic twist
	m_factor_after = 1.0 / (1.0+np.exp(-m_factor_before))

	# second layer
	array_ones = (np.array([np.ones(N)])).T
	m_factor_new = np.concatenate((m_factor_after, array_ones), axis=1)		# size_batch x (D+1)
	## tissue specific second layer
	Y_cellfactor_batch = []
	for i in range(len(list_tissue_batch)):
		k = list_tissue_batch[i]
		size_batch = len(Y_pos_batch[i])
		#
		Y_cellfactor = []
		beta_cellfactor2_reshape = beta_cellfactor2[k].T 										# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new[Y_pos_batch[i]], beta_cellfactor2_reshape)			# size_batch x J
		Y_cellfactor_batch.append(Y_cellfactor)
	Y_cellfactor_batch = np.array(Y_cellfactor_batch)



	##
	##
	## NOTE: extract out the cis- component
	##
	Y_cis_batch = []
	for i in range(len(list_tissue_batch)):
		k = list_tissue_batch[i]

		Y_cis_batch.append([])
		for pos_indiv in Y_pos_batch[i]:
			id = str(k) + '-' + str(pos_indiv)
			expr = repo_Y_cis_train[id]
			Y_cis_batch[i].append(expr)

		Y_cis_batch[i] = np.array(Y_cis_batch[i])
	Y_cis_batch = np.array(Y_cis_batch)
	##
	##
	##
	##




	##=============
	## compile and error cal
	##=============
	Y_final_batch = Y_cellfactor_batch + Y_cis_batch
	Tensor_error_batch = Y_final_batch - Y_batch









	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##==========================================================================================
	## backward prop
	##==========================================================================================
	##=========##=========##=========##=========##=========##=========##=========##=========##==
	N_sample = len(list_sample_batch)			## total number of samples (in this batch)

	##=============
	## from cell factor (for all tissues involved)
	##=============
	##== last layer
	for i in range(len(list_tissue_batch)):
		k = list_tissue_batch[i]
		der_cellfactor2[k] = np.zeros(beta_cellfactor2[k].shape)			# J x (D+1)
		m_factor_new_sub = m_factor_new[Y_pos_batch[i]]
		# J x N, N x (D+1)
		der_cellfactor2[k] = np.dot(Tensor_error_batch[i].T, m_factor_new_sub)
		der_cellfactor2[k] = der_cellfactor2[k] / N_sample

	##== first layer
	der_cellfactor1 = np.zeros(der_cellfactor1.shape)
	m_factor_der = np.multiply(m_factor_after, 1 - m_factor_after)
	for i in range(len(list_tissue_batch)):
		k = list_tissue_batch[i]
		# N x J, J x D --> N x D
		m_temp = np.dot(Tensor_error_batch[i], beta_cellfactor2[k][:, :-1])
		# N x D
		#m_factor_der = np.multiply(m_factor_after, 1 - m_factor_after)
		# N x D, N x D --> N x D
		m_temp = np.multiply(m_temp, m_factor_der[Y_pos_batch[i]])
		# D x N, N x (I+1)
		der_cellfactor1 += np.dot(m_temp.T, X[Y_pos_batch[i]])				# NOTE: this is too large for GPU --> we do the m_temp first, and then do Beta part by part
	der_cellfactor1 = der_cellfactor1 / N_sample



	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##==========================================================================================
	## regularization
	##==========================================================================================
	##=========##=========##=========##=========##=========##=========##=========##=========##==
	## especially for the beta_cellfactor1 and beta_cellfactor2
	rate_lasso_beta_cellfactor1 = 1.0
	sign = np.sign(beta_cellfactor1)
	der_cellfactor1 += rate_lasso_beta_cellfactor1 * sign
	#
	rate_lasso_beta_cellfactor2 = 1.0
	sign = np.sign(beta_cellfactor2)
	der_cellfactor2 += rate_lasso_beta_cellfactor2 * sign



	##=========##=========##=========##=========##=========##=========##=========##=========##==
	##==========================================================================================
	## gradient descent
	##==========================================================================================
	##=========##=========##=========##=========##=========##=========##=========##=========##==
	beta_cellfactor1 = beta_cellfactor1 - rate_learn * der_cellfactor1
	beta_cellfactor2 = beta_cellfactor2 - rate_learn * der_cellfactor2

	return






##==== calculate the total squared error for all tissues
def cal_error():
	global X, Y, Y_pos
	global beta_cellfactor1, beta_cellfactor2
	global I, J, K, D, N
	global Y_cis_train

	error_total = 0

	##================================================================================================================
	## cell factor first layer
	# first layer
	beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
	m_factor = np.dot(X, beta_cellfactor1_reshape)							# N x D
	# logistic twist
	m_factor = 1.0 / (1.0+np.exp(-m_factor))

	# second layer input
	array_ones = (np.array([np.ones(N)])).T
	m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)
	##================================================================================================================


	for k in range(K):

		##=============
		## from cell factor (tissue k)
		##=============
		Y_cellfactor = []
		beta_cellfactor2_reshape = beta_cellfactor2[k].T 						# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new, beta_cellfactor2_reshape)			# N x J

		##=============
		## compile and error cal
		##=============
		Y_final = Y_cellfactor
		list_pos = Y_pos[k]
		Y_final_sub = Y_final[list_pos]
		#error = np.sum(np.square(Y[k] - Y_final_sub))

		##
		##
		##
		##
		Y_cis_sub = Y_cis_train[k]
		error = np.sum(np.square(Y[k] - Y_cis_sub - Y_final_sub))
		##
		##
		##
		##


		##=============
		##=============
		error_total += error
		##=============
		##=============

	return error_total





def cal_error_test():
	global X_test, Y_test, Y_pos_test
	global beta_cellfactor1, beta_cellfactor2
	global I, J, K, D
	global Y_cis_test


	## make this N local
	N = len(X_test)
	X = X_test
	Y = Y_test
	Y_pos = Y_pos_test


	error_total = 0

	##================================================================================================================
	## cell factor first layer
	# first layer
	beta_cellfactor1_reshape = beta_cellfactor1.T 							# (I+1) x D
	m_factor = np.dot(X, beta_cellfactor1_reshape)							# N x D
	# logistic twist
	m_factor = 1.0 / (1.0+np.exp(-m_factor))

	# second layer input
	array_ones = (np.array([np.ones(N)])).T
	m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)
	##================================================================================================================


	for k in range(K):

		##=============
		## from cell factor (tissue k)
		##=============
		Y_cellfactor = []
		beta_cellfactor2_reshape = beta_cellfactor2[k].T 						# (D+1) x J
		Y_cellfactor = np.dot(m_factor_new, beta_cellfactor2_reshape)			# N x J

		##=============
		## compile and error cal
		##=============
		Y_final = Y_cellfactor
		list_pos = Y_pos[k]
		Y_final_sub = Y_final[list_pos]
		#error = np.sum(np.square(Y[k] - Y_final_sub))

		##
		##
		##
		##
		Y_cis_sub = Y_cis_test[k]
		error = np.sum(np.square(Y[k] - Y_cis_sub - Y_final_sub))
		##
		##
		##
		##

		##=============
		##=============
		error_total += error
		##=============
		##=============

	return error_total









##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============









if __name__ == "__main__":



	##==== get the training rate
	num_iter = int(sys.argv[1])
	rate_learn = float(sys.argv[2])
	print "num_iter is:", num_iter
	print "rate_learn is:", rate_learn



	print "now training..."



	##========================================================================
	## loading the data (real or simu)
	##========================================================================
	##==== load data
	##
	#fileheader = "../workbench1/data_simu_data/"
	fileheader = "../preprocess/data_train/"

	#
	X = np.load(fileheader + "X.npy")
	# Y and Y_pos
	K = 28										## TODO: specify the number of tissues
	Y = []
	Y_pos = []
	for k in range(K):
		data = np.load(fileheader + "Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load(fileheader + "Tensor_tissue_" + str(k) + "_pos.npy")
		Y.append(data)
		Y_pos.append(list_pos)
	Y = np.array(Y)
	Y_pos = np.array(Y_pos)

	##
	#fileheader = "../workbench1/data_simu_init/"
	#fileheader = "../preprocess/data_real_init/"
	fileheader = "../workbench6/data_real_init/"
	#
	beta_cellfactor1 = np.load(fileheader + "beta_cellfactor1.npy")
	beta_cellfactor2 = np.load(fileheader + "beta_cellfactor2.npy")
	##==== fill dimension
	I = len(X[0])
	J = len(Y[0][0])
	K = len(Y)
	N = len(X)
	D = len(beta_cellfactor1)

	#
	der_cellfactor1 = np.zeros(beta_cellfactor1.shape)
	der_cellfactor2 = np.zeros(beta_cellfactor2.shape)

	##==== append intercept to X, and Z (for convenience of cell factor pathway, and batch pathway)
	## X
	array_ones = (np.array([np.ones(N)])).T
	X = np.concatenate((X, array_ones), axis=1)									# N x (I+1)



	##========================================================================
	## loading the testing set
	##========================================================================
	##==== load data
	fileheader = "../preprocess/data_test/"

	#
	X_test = np.load(fileheader + "X.npy")
	# Y_test and Y_pos_test
	K = 28										## TODO: specify the number of tissues
	Y_test = []
	Y_pos_test = []
	for k in range(K):
		data = np.load(fileheader + "Tensor_tissue_" + str(k) + ".npy")
		list_pos = np.load(fileheader + "Tensor_tissue_" + str(k) + "_pos.npy")
		Y_test.append(data)
		Y_pos_test.append(list_pos)
	Y_test = np.array(Y_test)
	Y_pos_test = np.array(Y_pos_test)

	##==== append intercept to X_test (for convenience of cell factor pathway, and batch pathway)
	## X_test
	N_test = len(X_test)
	array_ones = (np.array([np.ones(N_test)])).T
	X_test = np.concatenate((X_test, array_ones), axis=1)						# N x (I+1)








	##==============================================
	## cis- pre-calculated components
	##==============================================
	Y_cis_train = np.load("../workbench54/data_real_init/Y_cis_train.npy")
	Y_cis_test = np.load("../workbench54/data_real_init/Y_cis_test.npy")
	#repo_Y_cis_train
	for k in range(len(Y_pos)):
		for i in range(len(Y_pos[k])):
			pos_indiv = Y_pos[k][i]
			id = str(k) + '-' + str(pos_indiv)
			repo_Y_cis_train[id] = Y_cis_train[k][i]








	##==============================================
	## cal the data variance (training and testing)
	##==============================================
	list_var_data = []
	# training set
	Y_flat = []
	for k in range(K):
		data = Y[k]
		data = data.tolist()
		Y_flat = Y_flat + data
	Y_flat = np.array(Y_flat)
	ave = np.mean(Y_flat, axis=0)
	var = np.sum(np.square(Y_flat - ave))
	list_var_data.append(var)
	print "training set total var:", var
	# testing set
	Y_flat = []
	for k in range(K):
		data = Y_test[k]
		data = data.tolist()
		Y_flat = Y_flat + data
	Y_flat = np.array(Y_flat)
	ave = np.mean(Y_flat, axis=0)
	var = np.sum(np.square(Y_flat - ave))
	list_var_data.append(var)
	print "testing set total var:", var
	# save
	np.save("./result/list_var_data", list_var_data)










	##=========================================
	## prepare for the stochastic sample pool
	##=========================================
	##
	## NOTE: this is the key step to make SGD unbiased
	##
	list_sample = []
	for k in range(len(Y_pos)):
		for n in range(len(Y_pos[k])):
			sample = str(k) + "-" + str(n)
			list_sample.append(sample)
	list_sample = np.array(list_sample)









	##============
	## train
	##============
	##==== timer, for speed test
	start_time_total = timeit.default_timer()

	list_error = []
	list_error_test = []
	for iter1 in range(num_iter):
		print "[@@@]working on out iter#", iter1


		##==== timer
		start_time = timeit.default_timer()



		if iter1 == 0:
			## error before
			##============================================
			error = cal_error()
			print "[error_before] current total error (train):", error
			list_error.append(error)

			error = cal_error_test()
			print "[error_before] current total error (test):", error
			list_error_test.append(error)
			##============================================



		forward_backward_gd()



		## error after
		##============================================
		error = cal_error()
		print "[error_after] current total error (train):", error
		list_error.append(error)
		np.save("./result/list_error", np.array(list_error))

		error = cal_error_test()
		print "[error_after] current total error (test):", error
		list_error_test.append(error)
		np.save("./result/list_error_test", np.array(list_error_test))
		##============================================


		##==== timer
		elapsed = timeit.default_timer() - start_time
		print "time spent on this batch:", elapsed




		##==== save results per need
		if iter1 % 10 == 0:
			start_time = timeit.default_timer()
			np.save("./result/beta_cellfactor1", beta_cellfactor1)
			np.save("./result/beta_cellfactor2", beta_cellfactor2)
			elapsed = timeit.default_timer() - start_time
			print "time spent on saving the data:", elapsed




	print "done!"
	##==== timer, for speed test
	print "speed:", (timeit.default_timer() - start_time_total) / num_iter


	##==== save the model
	np.save("./result/beta_cellfactor1", beta_cellfactor1)
	np.save("./result/beta_cellfactor2", beta_cellfactor2)
	print "now it's done..."












