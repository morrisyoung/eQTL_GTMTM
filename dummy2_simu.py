import numpy as np
import math





##==== scale of the input data (real data)
I = 24451
J = 19425
D = 100
N = 450







if __name__ == "__main__":




	# X
	X = np.random.random_sample((N, I)) * 2

	# beta
	beta1 = np.random.normal(size=(D, I+1)) 						# NOTE: intercept
	beta2 = np.random.normal(size=(J, D+1)) 						# NOTE: intercept

	# Y
	# first layer
	array_ones = (np.array([np.ones(N)])).T
	X_new = np.concatenate((X, array_ones), axis=1)							# N x (I+1)
	beta1_reshape = beta1.T 												# (I+1) x D
	m_factor = np.dot(X_new, beta1_reshape)									# N x D
	# second layer
	array_ones = (np.array([np.ones(N)])).T
	m_factor_new = np.concatenate((m_factor, array_ones), axis=1)			# N x (D+1)
	beta2_reshape = beta2.T 												# (D+1) x J
	Y = np.dot(m_factor_new, beta2_reshape)									# N x J





	##==== save data
	np.save("./data_simu/X", X)
	np.save("./data_simu/Y", Y)
	np.save("./data_simu/beta1_real", beta1)
	np.save("./data_simu/m_factor_real", m_factor)
	np.save("./data_simu/beta2_real", beta2)





	##====================================================
	## simu another copy as the init (randomly use another copy to init -- we can of course init more wisely)
	##====================================================
	beta1 = np.random.normal(size=(D, I+1)) 						# NOTE: intercept
	beta2 = np.random.normal(size=(J, D+1)) 						# NOTE: intercept
	np.save("./data_simu/beta1_init", beta1)
	np.save("./data_simu/beta2_init", beta2)













