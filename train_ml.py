import numpy as np
import tensorflow as tf
import timeit











##==================================================================================================================
X = np.load("./data_real/X.npy")
# add the intercept to X:
m_ones = np.ones((len(X), 1))
X = np.concatenate((X, m_ones), axis=1)									# N x (I+1)

Y = np.load("./data_real/Y_spread.npy")						## train part (indexing); test part (indexing); Nan part
beta1_init = np.load("./data_real/beta1_init.npy")			## NOTE: let's make the shape of them ready
beta2_init = np.load("./data_real/beta2_init.npy")			## NOTE: let's make the shape of them ready
m_factor_init = np.load("./data_real/m_factor_init.npy")

list_index_sample_train = np.load("./data_real/list_index_sample_train.npy")
list_index_sample_test = np.load("./data_real/list_index_sample_test.npy")












with tf.device("/cpu:0"):





	##==================================================================================================================
	## data and model
	x = tf.placeholder(tf.float32, shape=(None, len(X[0])))
	#y = tf.placeholder(tf.float32, shape=(None, len(Y[0])))



	## beta1
	place_beta1 = tf.placeholder(tf.float32, shape=(len(beta1_init), len(beta1_init[0])))
	beta1 = tf.Variable(place_beta1)

	## m_factor
	place_m_factor = tf.placeholder(tf.float32, shape=(len(m_factor_init), len(m_factor_init[0])))
	m_factor = tf.Variable(place_m_factor)

	## expand m_factor
	tensor_constant = tf.constant(1.0, dtype=tf.float32, shape=[len(m_factor_init), 1])
	m_factor_ext = tf.concat([m_factor, tensor_constant], 1)

	## beta2
	place_beta2 = tf.placeholder(tf.float32, shape=(len(beta2_init), len(beta2_init[0]), len(beta2_init[0][0])))
	beta2 = tf.Variable(place_beta2)




	##
	f = tf.matmul(x, beta1)

 	## tensordot: (indiv, factor+1) x (tissue, factor+1, gene) = (indiv, tissue, gene)
	result_exp = tf.tensordot(m_factor_ext, beta2, [[1], [1]])
	result_exp_reshape = tf.transpose(result_exp, perm=[1, 0, 2])
	result_exp_flatten = tf.reshape(result_exp_reshape, [-1])
	y_index = tf.placeholder(tf.int32)
	y_ = tf.gather(result_exp_flatten, y_index)

	# real Y
	y = tf.placeholder(tf.float32)









	##==================================================================================================================
	## prediction cost
	cost_y = tf.reduce_sum(tf.square(tf.subtract(y_, y)))


	## factor cost --> the prior for factor --> genetic cost
	cost_m_factor = tf.reduce_sum(tf.square(tf.subtract(f, m_factor)))


	## sparsity (it seems we also penalize the intercept)
	coeff_regularizer = tf.constant(.001)
	norm_sums = tf.add(tf.reduce_sum(tf.abs(beta1)),
	                   tf.reduce_sum(tf.abs(beta2)))
	cost_regularizer = tf.multiply(coeff_regularizer, norm_sums)


	## total train cost
	cost_train = tf.add(cost_y, cost_m_factor)
	cost_train = tf.add(cost_train, cost_regularizer)


	##
	lr = tf.constant(0.000000000005, name='learning_rate')
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)


	## learn!!!
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_step1 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta1])
	training_step2 = optimizer.minimize(cost_train, global_step=global_step, var_list=[m_factor])
	training_step3 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta2])









	##==================================================================================================================
	# execute
	#init = tf.initialize_all_variables()
	init = tf.global_variables_initializer()
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	sess = tf.Session()
	#sess.run(init)
	sess.run(init, feed_dict={place_beta1: beta1_init, place_beta2: beta2_init, place_m_factor: m_factor_init})





	list_error_train = []
	list_error_test = []
	for i in xrange(1000):
		print "iter#", i,
		#### train step
		sess.run(training_step1, feed_dict={x: X, y_index: list_index_sample_train, y: Y[list_index_sample_train]})
		sess.run(training_step2, feed_dict={x: X, y_index: list_index_sample_train, y: Y[list_index_sample_train]})
		sess.run(training_step3, feed_dict={x: X, y_index: list_index_sample_train, y: Y[list_index_sample_train]})



		#### training error
		error = sess.run(cost_y, feed_dict={x: X, y_index: list_index_sample_train, y: Y[list_index_sample_train]})
		print "training error:", error,
		list_error_train.append(error)
		np.save("./result/list_error_train", list_error_train)


		#### testing error
		error = sess.run(cost_y, feed_dict={x: X, y_index: list_index_sample_test, y: Y[list_index_sample_test]})
		print "testing error:", error
		list_error_test.append(error)
		np.save("./result/list_error_test", list_error_test)

















