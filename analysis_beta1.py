import numpy as np
import matplotlib.pyplot as plt






if __name__ == "__main__":





	##==== extract single facctors (without the mean effects)
	##
	"""
	data = np.load("./result/beta_cellfactor1.npy")
	print data.shape


	for i,array in enumerate(data):
		print i
		np.save("./result_processed/beta_cellfactor1_f" + str(i), array[:-1])
	"""



	##
	"""
	data = np.load("./data_real_init/beta_cellfactor1.npy")
	print data.shape


	for i,array in enumerate(data):
		print i
		np.save("./data_real_init_processed/beta_cellfactor1_init_f" + str(i), array[:-1])
	"""






	##====================================================================================================================
	## test learning strength
	##====================================================================================================================
	d = 6
	beta_d_init = np.load("./result/beta_cellfactor1_init_f" + str(d) + ".npy")
	beta_d_learned = np.load("./result/beta_cellfactor1_f" + str(d) + ".npy")



	beta_d_diff = beta_d_learned - beta_d_init
	plt.plot(beta_d_init, beta_d_diff, 'ro', alpha=0.5)
	#list = np.arange(-0.05, 0.25, 0.001)
	#plt.plot(list, list, 'b-', alpha=0.5)
	plt.grid()
	plt.xlabel("beta of init")
	plt.ylabel("beta of learned - beta of init")
	plt.title("d=" + str(d))
	plt.show()
















