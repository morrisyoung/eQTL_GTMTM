import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines









if __name__=="__main__":





	# list_error = np.load("./result/list_error_train.npy")
	# plt.grid(True)
	# plt.plot(list_error)
	# plt.show()

	# list_error_train = np.load("./result/list_error_train.npy")
	# list_error_test = np.load("./result/list_error_test.npy")
	# plt.plot(list_error_train, 'r')
	# plt.plot(list_error_test, 'b')
	# plt.grid(True)
	# plt.show()

	plt.figure(1)
	list_error_train = np.load("./result/list_error_train.npy")
	list_error_test = np.load("./result/list_error_test.npy")
	plt.subplot(121)
	plt.plot(list_error_train[:100], 'r')
	plt.grid(True)
	plt.subplot(122)
	plt.plot(list_error_test[:100], 'b')
	plt.grid(True)
	plt.show()













