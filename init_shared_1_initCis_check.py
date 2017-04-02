import numpy as np
import math





## check the variance explained in the whole dataset





if __name__ == "__main__":



	"""
	Y = np.load("./data_raw/Data_final.npy")
	Y_cis_all = np.load("./data_raw/Y_cis_all.npy")
	print Y_cis_all.shape


	Y_spread = []
	Y_cis_all_spread = []
	for k in range(len(Y)):
		for n in range(len(Y[k])):
			if not math.isnan(Y[k][n][0]):
				Y_spread.append(Y[k][n])
				Y_cis_all_spread.append(Y_cis_all[k][n])
	print len(Y_spread)
	Y_spread = np.array(Y_spread)
	Y_cis_all = np.array(Y_cis_all)


	Y_spread_ave = np.mean(Y_spread, axis=0)
	var_total = np.sum(np.square(Y_spread - Y_spread_ave))
	var_cis = np.sum(np.square(Y_spread - Y_cis_all_spread))
	print "var_cis / var_total:", var_cis / var_total
	"""













