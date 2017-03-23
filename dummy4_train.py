import numpy as np
import tensorflow as tf
import timeit







## will do the iterative approach for graphical model, with training and testing procedures

## use the data simu and init from dummy2










##==================================================================================================================
X = np.load("./data_simu/X.npy")
Y = np.load("./data_simu/Y.npy")
beta1_init = np.load("./data_simu/beta1_init.npy")
beta2_init = np.load("./data_simu/beta2_init.npy")
beta1_init = beta1_init.T
beta2_init = beta2_init.T
m_factor_init = np.load("./data_simu/m_factor_init.npy")



list_index_sample_train = np.load("./data_simu/list_index_sample_train.npy")
list_index_sample_test = np.load("./data_simu/list_index_sample_test.npy")



# spread Y
Y_spread = np.reshape(Y, -1)








def get_indiv_list(list_sample):
	list_indiv = []
	for index in list_sample:
		indiv = index / len(Y[0])
		list_indiv.append(indiv)
	return list_indiv







with tf.device("/cpu:0"):






	##==================================================================================================================
	## data and model
	x = tf.placeholder(tf.float32, shape=(None, len(X[0])))
	#y = tf.placeholder(tf.float32, shape=(None, len(Y[0])))

	# beta1
	place_beta1 = tf.placeholder(tf.float32, shape=(len(beta1_init)-1, len(beta1_init[0])))
	beta1 = tf.Variable(place_beta1)
	place_beta1_intercept = tf.placeholder(tf.float32, shape=(len(beta1_init[0])))
	beta1_intercept = tf.Variable(place_beta1_intercept)
	# m_factor
	place_m_factor = tf.placeholder(tf.float32, shape=(len(m_factor_init), len(m_factor_init[0])))
	m_factor = tf.Variable(place_m_factor)
	# beta2
	place_beta2 = tf.placeholder(tf.float32, shape=(len(beta2_init)-1, len(beta2_init[0])))
	beta2 = tf.Variable(place_beta2)
	place_beta2_intercept = tf.placeholder(tf.float32, shape=(len(beta2_init[0])))
	beta2_intercept = tf.Variable(place_beta2_intercept)

	#
	f = tf.matmul(x, beta1) + beta1_intercept
	# no activation layer in the middle
	#y_ = tf.matmul(m_factor, beta2) + beta2_intercept
	result_exp = tf.matmul(m_factor, beta2) + beta2_intercept
	result_exp_flatten = tf.reshape(result_exp, [-1])
	y_index = tf.placeholder(tf.int32)
	y_ = tf.gather(result_exp_flatten, y_index)

	# real Y
	y = tf.placeholder(tf.float32)







	##==================================================================================================================
	## prediction cost
	cost_y = tf.reduce_sum(tf.square(tf.subtract(y_, y)))


	## factor cost --> the prior for factor --> genetic cost
	x_index = tf.placeholder(tf.int32)
	f_sub = tf.gather(f, x_index)
	m_factor_sub = tf.gather(m_factor, x_index)
	cost_m_factor = tf.reduce_sum(tf.square(tf.subtract(f_sub, m_factor_sub)))


	## sparsity
	coeff_regularizer = tf.constant(.001)
	norm_sums = tf.add(tf.reduce_sum(tf.abs(beta1)),
	                   tf.reduce_sum(tf.abs(beta2)))
	cost_regularizer = tf.multiply(coeff_regularizer, norm_sums)


	## total train cost
	cost_train = tf.add(cost_y, cost_m_factor)
	cost_train = tf.add(cost_train, cost_regularizer)


	##
	#lr = tf.constant(0.00000000008, name='learning_rate')				# for random init
	#lr = tf.constant(0.00000000001, name='learning_rate')				# for random init


	#lr = tf.constant(0.000000000008, name='learning_rate')
	#lr = tf.constant(0.000000000005, name='learning_rate')
	lr = tf.constant(0.000000000003, name='learning_rate')




	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)


	## learn!!!
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_step = optimizer.minimize(cost_train, global_step=global_step)








	##==================================================================================================================
	# execute
	init = tf.initialize_all_variables()
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	sess = tf.Session()
	#sess.run(init)
	sess.run(init, feed_dict={place_beta1: beta1_init[:-1, :],
							place_beta1_intercept: beta1_init[-1, :],
							place_beta2: beta2_init[:-1,:],
							place_beta2_intercept: beta2_init[-1,:],
							place_m_factor: m_factor_init}
							)

	list_error_train = []
	list_error_test = []
	for i in xrange(10000):
		print "iter#", i,
		## mini-batch training
		np.random.shuffle(list_index_sample_train)
		size_batch = 20
		list_index_sample_train_batch = list_index_sample_train[:size_batch]
		list_index_indiv_train_batch = get_indiv_list(list_index_sample_train_batch)
		#
		sess.run(training_step, feed_dict={x: X,
											x_index: list_index_indiv_train_batch,
											y: Y_spread[list_index_sample_train_batch],
											y_index: list_index_sample_train_batch})



		## training error
		error = sess.run(cost_y, feed_dict={y: Y_spread[list_index_sample_train], y_index: list_index_sample_train})
		print "train error:", error,
		list_error_train.append(error)
		np.save("./result/list_error_train", list_error_train)

		## testing error
		error = sess.run(cost_y, feed_dict={y: Y_spread[list_index_sample_test], y_index: list_index_sample_test})
		print "test error:", error
		list_error_test.append(error)
		np.save("./result/list_error_test", list_error_test)
















