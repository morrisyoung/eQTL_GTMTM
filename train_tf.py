import numpy as np
import tensorflow as tf
import timeit








##
## Mar.18, 17: to adapt this code
##









## below are some reference code (for Tensor Product):
## can use code structure from train_ml.py


with tf.device("/cpu:0"):




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













