import numpy as np
import math







##==== scale of the input data (real data)
I = 24451
J = 19425
D = 100
N = 450
K = 2








if __name__ == "__main__":




	# X
	X = np.random.random_sample((N, I)) * 2


	# beta
	beta1 = np.random.normal(size=(I+1, D))
	beta2 = np.random.normal(size=(K, D+1, J))


	# Y
	# first layer
	m_ones = np.ones((N, 1))
	X_ext = np.concatenate((X, m_ones), axis=1)								# N x (I+1)
	m_factor = np.dot(X_ext, beta1)											# N x D
	# second layer
	m_ones = np.ones((N, 1))
	m_factor_ext = np.concatenate((m_factor, m_ones), axis=1)				# N x (D+1)
	# (indiv, factor+1) x (tissue, factor+1, gene) = (indiv, tissue, gene)
	Y = np.tensordot(m_factor_ext, beta2, axes=([1],[1]))
	Y = np.transpose(Y, (1, 0, 2))
	Y_spread = np.reshape(Y, -1)






	##==== save data
	np.save("./data_real/X", X)
	np.save("./data_real/Y", Y)
	np.save("./data_real/Y_spread", Y_spread)
	np.save("./data_real/beta1_real", beta1)
	np.save("./data_real/beta2_real", beta2)






	##====================================================
	## simu another copy as the init (randomly use another copy to init -- we can of course init more wisely)
	##====================================================
	beta1 = np.random.normal(size=(I+1, D))
	beta2 = np.random.normal(size=(K, D+1, J))
	np.save("./data_real/beta1_init", beta1)
	np.save("./data_real/beta2_init", beta2)















