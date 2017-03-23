import numpy as np
import tensorflow as tf
import pandas as pd
import timeit








## Mar.18, 17: to adapt this code











#### code for Genetic Tensor Decomposition (GTD), with Stochastic Gradient Descent (SGD) algorithm
#### pick individual wiht all his/her samples (of all tissues), with probability propotional to pool size (for this indiv)
#### (Feb.15) NOTE: the above sampling scheme is not correct; we still need to sample uniformly from all samples (with mini-batch)










##==================================================================================================================
'''
##
T = np.load("./data_simu_gtd/T.npy")
U = np.load("./data_simu_gtd/U.npy")
V = np.load("./data_simu_gtd/V.npy")
##
Beta = np.load("./data_simu_gtd/Beta.npy")
##
#Y = np.load("./data_simu_gtd/Y.npy")
Y_spread = np.load("./data_simu_gtd/Y_spread.npy")					## this for now is a full tensor
X = np.load("./data_simu_gtd/X.npy")
##
table_index_indiv = np.load("./data_simu_gtd/table_index_indiv.npy")
pool_index_indiv = {}
for i in range(len(table_index_indiv)):
	pool_index_indiv[i] = table_index_indiv[i]
list_index_all = np.load("./data_simu_gtd/list_index_train.npy")


## for categorical draw:
list_p = np.load("./data_simu_gtd/list_p_indiv.npy")
'''

'''
#### real data loader
##
T = np.load("./data_real_init/fm_tissue.npy")
U = np.load("./data_real_init/fm_indiv.npy")
V = np.load("./data_real_init/fm_gene.npy")
##
Beta = np.load("./data_real_init/Beta.npy")
##
Y_spread = np.load("./data_real_init/Y_spread.npy")					## this for now is a full tensor
X = np.load("./data_real_init/X.npy")
array_ones = np.array([np.ones(len(X))]).T
X = np.concatenate((X, array_ones), axis=1)

##
table_index_indiv = np.load("./data_real_init/table_index_indiv.npy")
pool_index_indiv = {}
for i in range(len(table_index_indiv)):
	pool_index_indiv[i] = table_index_indiv[i]
list_index_all = np.load("./data_real_init/list_index_all.npy")


## for categorical draw:
list_p = np.load("./data_real_init/list_p_indiv.npy")

## for accessing the pos of samples in the incomplete tensor -- (k, indiv)
list_pos_all = np.load("./data_real_init/list_pos_all.npy")
'''









##
dimension1 = len(T)
dimension2 = len(U)
dimension3 = len(V)
feature_len = len(T[0])








with tf.device("/cpu:0"):




	#### the genetic component
	placeholder_index_x = tf.placeholder(tf.int32)
	x = tf.placeholder(tf.float32, shape=(None, len(X[0])))							## genotype
	place_beta = tf.placeholder(tf.float32, shape=(len(Beta), len(Beta[0])))
	beta = tf.Variable(place_beta)
	u_ = tf.matmul(x, beta)


	#### the tensor product
	## for T, we have the extra dimension for broadcasting the multiply op
	'''
	T = tf.Variable(initial_value=tf.truncated_normal([dimension1, 1, feature_len]), name='tissues')
	U = tf.Variable(initial_value=tf.truncated_normal([dimension2, feature_len]), name='indivs')
	V = tf.Variable(initial_value=tf.truncated_normal([dimension3, feature_len]), name='genes')
	'''
	#### initialize the three with pre-loaded fm
	T = tf.Variable(initial_value=T, dtype=tf.float32, name='tissues')
	U = tf.Variable(initial_value=U, dtype=tf.float32, name='indivs')
	V = tf.Variable(initial_value=V, dtype=tf.float32, name='genes')
	T = tf.expand_dims(T, 1)



	TUD = tf.multiply(T, U, name=None)					## dimension1 x dimension2 x feature_len
	result = tf.einsum('kid,jd->kij', TUD, V)			## dimension1 x dimension2 x dimension3
	result_flatten = tf.reshape(result, [-1])

	# expected Y (inside Gene Tensor)
	placeholder_index_y = tf.placeholder(tf.int32)
	y_ = tf.gather(result_flatten, placeholder_index_y)

	# real Y
	y = tf.placeholder(tf.float32)


	####
	#### SO: to fill in this batch the following feed dic:
	####	placeholder_index_x (to make it a list, to make it general to multiple individuals)
	####	x
	####	placeholder_index_y
	####	y
	####




	##==================================================================================================================
	## cost function
	base_cost = tf.reduce_sum(tf.square(tf.subtract(y_, y)))


	## the prior for U --> genetic cost
	U_sub = tf.gather(U, placeholder_index_x)
	U_cost = tf.reduce_sum(tf.square(tf.subtract(u_, U_sub)))


	## the prior for V and T --> regularization (for V and T)
	lda_VT = tf.constant(.001)
	norm_sums = tf.add(tf.reduce_sum(tf.abs(T)),
	                   tf.reduce_sum(tf.abs(V)))
	regularizer_VT = tf.multiply(norm_sums, lda_VT)


	## regularization (for Beta)
	lda_beta = tf.constant(.001)
	norm_sums_beta = tf.reduce_sum(tf.abs(beta))
	regularizer_beta = tf.multiply(norm_sums_beta, lda_beta)


	## total train cost
	cost_train = tf.add(base_cost, U_cost)
	cost_train = tf.add(cost_train, regularizer_VT)
	cost_train = tf.add(cost_train, regularizer_beta)


	## learning rate
	## 0.00000001 works once, but seems a little slow
	## 0.00000005 still goes wild

	lr = tf.constant(0.00000003, name='learning_rate')
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)


	## learn!!!
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_step = optimizer.minimize(cost_train, global_step=global_step)






	##==================================================================================================================
	# execute
	init = tf.initialize_all_variables()
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	#sess = tf.Session()
	#sess.run(init)
	sess.run(init, feed_dict={place_beta: Beta})



	list_error = []

	##==== timer
	start_time = timeit.default_timer()
	for i in xrange(1000):


		#### sample individuals based on tissues avaliable for that individual (unbiased sampling for loss function)
		# list_temp = np.random.multinomial(1, list_p)
		# index = np.argmax(list_temp)
		# list_index_x = [index]
		# # pick up sample indices (in the spread version of Y) for that individual
		# list_index_y = pool_index_indiv[index]
		# # call run and feed in data
		# sess.run(training_step, feed_dict={placeholder_index_x: list_index_x, x: [X[index]], placeholder_index_y: list_index_y, y: Y_spread[list_index_y]})


		## Feb.14: still need to sample from incomplete tensor, rather than gathering samples from each individual, to make it unbiased in the loss func (loglike)
		N_sample = len(list_index_all) / dimension3
		#print "there are totally",
		#print N_sample,
		#print "samples, and we are sampling mini-batch from them uniformly randomly..."
		#
		size_batch = 20
		list_temp = np.random.permutation(N_sample)[:size_batch]
		#
		list_index_x = list_pos_all[list_temp][:,1]
		# pick up sample indices (in the spread version of Y) for that individual
		list_index_y = []
		for pos in list_temp:
			list_index_y += list_index_all[pos*dimension3: pos*dimension3+dimension3].tolist()
		list_index_y = np.array(list_index_y)
		# call run and feed in data
		sess.run(training_step, feed_dict={placeholder_index_x: list_index_x, x: X[list_index_x], placeholder_index_y: list_index_y, y: Y_spread[list_index_y]})



		####
		## training error
		N = len(X)
		list_index_x = np.arange(N)
		error = sess.run(cost_train, feed_dict={placeholder_index_x: list_index_x, x: X, placeholder_index_y: list_index_all, y: Y_spread[list_index_all]})
		print error
		list_error.append(error)
		np.save("./result/list_error", list_error)



		####
		## save fm per 50 iters
		if i % 50 == 0:
			fm_tissue = T.eval(session=sess)
			fm_indiv = U.eval(session=sess)
			fm_gene = V.eval(session=sess)
			np.save("./result/" + str(i) + "_T", fm_tissue)
			np.save("./result/" + str(i) + "_U", fm_indiv)
			np.save("./result/" + str(i) + "_V", fm_gene)




	##==== timer
	elapsed = timeit.default_timer() - start_time
	print "time spent:", elapsed













