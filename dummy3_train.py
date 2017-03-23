import numpy as np
import tensorflow as tf
import timeit





## still back to do the redundant linear model, without modeling the hidden factors as hidden variables
## we try to use an iterative approach to optimize the objective function
## will do this for the incomplete tensor setting



## dummy3 is trying the direct linear model; now from just random init







##==================================================================================================================
X = np.load("./data_simu/X.npy")
Y = np.load("./data_simu/Y.npy")
beta1_init = np.load("./data_simu/beta1_init.npy")
beta2_init = np.load("./data_simu/beta2_init.npy")
beta1_init = beta1_init.T
beta2_init = beta2_init.T


threshold = 335
X_test = X[threshold:]
Y_test = Y[threshold:]
X = X[:threshold]
Y = Y[:threshold]










with tf.device("/cpu:0"):





	##==================================================================================================================
	## data and model
	x = tf.placeholder(tf.float32, shape=(None, len(X[0])))
	y = tf.placeholder(tf.float32, shape=(None, len(Y[0])))




	# beta1 -> turn into slope and intercept
	place_beta1 = tf.placeholder(tf.float32, shape=(len(beta1_init), len(beta1_init[0])))
	beta1 = tf.Variable(place_beta1)
	list_index_beta1_slope = np.arange(len(beta1_init))[:-1]
	beta1_slope = tf.gather(beta1, list_index_beta1_slope)
	beta1_intercept = tf.gather(beta1, len(beta1_init)-1)
	# beta2 -> turn into slope and intercept
	place_beta2 = tf.placeholder(tf.float32, shape=(len(beta2_init), len(beta2_init[0])))
	beta2 = tf.Variable(place_beta2)
	list_index_beta2_slope = np.arange(len(beta2_init))[:-1]
	beta2_slope = tf.gather(beta2, list_index_beta2_slope)
	beta2_intercept = tf.gather(beta2, len(beta2_init)-1)


	#
	f = tf.matmul(x, beta1_slope) + beta1_intercept
	# no activation layer in the middle
	y_ = tf.matmul(f, beta2_slope) + beta2_intercept





	##==================================================================================================================
	## cost function
	cost_base = tf.reduce_sum(tf.square(tf.subtract(y_, y)))


	## sparsity
	coeff_regularizer = tf.constant(.001)
	norm_sums = tf.add(tf.reduce_sum(tf.abs(beta1)),
	                   tf.reduce_sum(tf.abs(beta2)))
	cost_regularizer = tf.multiply(coeff_regularizer, norm_sums)


	## total train cost
	cost_train = tf.add(cost_base, cost_regularizer)


	##
	lr = tf.constant(0.000000000005, name='learning_rate')
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)


	## learn!!!
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#training_step1 = optimizer.minimize(cost_train, global_step=global_step)
	training_step1 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta1])
	training_step2 = optimizer.minimize(cost_train, global_step=global_step, var_list=[beta2])





	##==================================================================================================================
	# execute
	init = tf.initialize_all_variables()
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	sess = tf.Session()
	#sess.run(init)
	sess.run(init, feed_dict={place_beta1: beta1_init, place_beta2: beta2_init})




	list_error_train = []
	list_error_test = []
	for i in xrange(1000):
		print "iter#", i
		sess.run(training_step1, feed_dict={x: X, y: Y})
		sess.run(training_step2, feed_dict={x: X, y: Y})




		## training error
		#error = sess.run(cost_train, feed_dict={x: X, y: Y})
		error = sess.run(cost_base, feed_dict={x: X, y: Y})
		print "train error:", error,
		list_error_train.append(error)
		np.save("./result/list_error_train", list_error_train)


		## testing error
		#error = sess.run(cost_train, feed_dict={x: X, y: Y})
		error = sess.run(cost_base, feed_dict={x: X_test, y: Y_test})
		print "test error:", error
		list_error_test.append(error)
		np.save("./result/list_error_test", list_error_test)














