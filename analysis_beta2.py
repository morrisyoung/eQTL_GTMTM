import numpy as np
import matplotlib.pyplot as plt





## for NN and ML, we pick up each tissue to check
## for TM, we check gene fm and tissue fm









if __name__ == "__main__":








	##====================================================================================================================
	## test learning strength
	##====================================================================================================================
	beta_init = np.load("./result/beta_cellfactor2_init.npy")
	beta_learned = np.load("./result/beta_cellfactor2_learned.npy")


	## tissue specific
	k = 6
	beta_k_d_init = beta_init[k, :, :-1]
	beta_k_d_learned = beta_learned[k, :, :-1]



	# #
	# beta_k_d_init = beta_init[:, :, :-1]
	# beta_k_d_learned = beta_learned[:, :, :-1]
	# #
	# beta_k_d_init = beta_k_d_init.ravel()
	# beta_k_d_learned = beta_k_d_learned.ravel()










	beta_k_d_diff = beta_k_d_learned - beta_k_d_init
	plt.plot(beta_k_d_init, beta_k_d_diff, 'ro', alpha=0.5)
	#list = np.arange(-0.05, 0.25, 0.001)
	#plt.plot(list, list, 'b-', alpha=0.5)
	plt.grid()
	plt.xlabel("beta of init")
	plt.ylabel("beta of learned - beta of init")
	#plt.title("k=" + str(k) + ", d=" + str(d))
	#plt.title("k=" + str(k))
	plt.show()















