import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines






list_tissue_color = ['k', '#988ED5', 'm', '#8172B2', '#348ABD', '#EEEEEE', '#FF9F9A', '#56B4E9', '#8C0900', '#6d904f', 'cyan', 'red', 'g']
list_tissue = ['tissue#0', 'tissue#1', 'tissue#2', 'tissue#3', 'tissue#4', 'tissue#5', 'tissue#6', 'tissue#7', 'tissue#8', 'tissue#9', 'tissue#10', 'tissue#11', 'tissue#12']







def load_array_txt(filename):

	array = []
	file = open(filename, 'r')
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		value = float(line)
		array.append(value)
	file.close()
	array = np.array(array)

	return array






if __name__=="__main__":




	'''
	##==== total likelihood
	arr = np.load("./result/list_error.npy")
	arr1 = np.load("./result/list_error_test.npy")
	#print arr
	print "len of error list:",
	print len(arr)
	#print arr[-10:]
	print arr[:10]
	print arr[-1]
	#print len(list_tissue_color)
	#print len(list_tissue)
	plt.plot(arr, 'r-')
	plt.plot(arr1, 'b-')
	plt.xlabel("number of batches")
	plt.ylabel("total squared error")
	plt.title("total squared error v.s. num of batches")
	plt.grid()
	## the total variance of samples for this simulation set
	#plt.plot([1489807752.51]*len(arr), 'b-')
	'''








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
	plt.plot(list_error_train[:], 'r')
	plt.grid(True)
	plt.subplot(122)
	plt.plot(list_error_test[:], 'b')
	plt.grid(True)
	plt.show()








	"""
	## NOTE: plot double legend
	#x = np.arange(0, 201, 1)
	#y1 = load_array_txt("./result/error_total_online.txt")
	#y2 = load_array_txt("./result/error_total_online_test.txt")
	y1 = np.load("./result/list_error.npy")
	y2 = np.load("./result/list_error_test.npy")
	x = np.arange(0, len(y1), 1)

	fig, ax1 = plt.subplots()

	#ax2 = ax1.twinx()


	## TEST
	print y1
	print y2


	## scale to variance portion
	y1 = 1 - y1 / 82944750.0					# amount: (4270, 19425)
	y2 = 1 - y2 / 26776502.0985					# amount: (1424, 19425)


	## TEST
	print y1
	print y2
	print len(y1)
	print len(y2)


	## TEST
	print "training VE first and last:", y1[0], y1[-1]
	print "testing VE first and last:", y2[0], y2[-1]



	ax1.plot(x, y1, 'b-')
	ax1.plot(x, y2, 'g-')
	#ax2.plot(x, y2, 'g-', label="Testing")

	ax1.set_xlabel('# of updates')
	ax1.set_ylabel('variance explained')
	#x2.set_ylabel('testing total squared error', color='g')

	blue_line = mlines.Line2D([], [], color='b', markersize=15, label='training set')
	red_line = mlines.Line2D([], [], color='g', markersize=15, label='testing set')
	plt.legend(handles=[blue_line, red_line], loc = 4)


	#plt.axis([0, len(y1), 0, 1])
	plt.grid(True)
	plt.title("variance explained by the model v.s. learning updates")
	plt.show()
	"""










	"""
	##===================
	plt.subplot(121)
	arr1 = np.load("./result/list_error_1.npy")
	plt.plot(arr1, 'r-')
	arr2 = np.load("./result/list_error_2.npy")
	plt.plot(arr2, 'g-')
	arr3 = np.load("./result/list_error_3.npy")
	plt.plot(arr3, 'b-')
	#arr = np.load("./result/list_error_4.npy")
	#plt.plot(arr, 'y-')
	list_handle = []
	line = mlines.Line2D([], [], color='r', label='rate1')
	list_handle.append(line)
	line = mlines.Line2D([], [], color='g', label='10*rate1')
	list_handle.append(line)
	line = mlines.Line2D([], [], color='b', label='100*rate1')
	list_handle.append(line)
	plt.legend(handles=list_handle)
	plt.xlabel("number of batches")
	plt.ylabel("total squared error")
	plt.title("total squared error v.s. num of batches")
	plt.grid()
	##===================
	plt.subplot(122)
	plt.plot(arr1, 'r-')
	list_handle = []
	line = mlines.Line2D([], [], color='r', label='rate1')
	list_handle.append(line)
	plt.legend(handles=list_handle)
	plt.xlabel("number of batches")
	plt.ylabel("total squared error")
	plt.title("total squared error v.s. num of batches")
	plt.grid()
	"""








	"""
	##==== plot per tissue errir curve
	arr_sub = []
	K = 13
	num_iter_in = 100
	pos = num_iter_in*3			# 0 range to 12
	while pos+num_iter_in<=len(arr):
		temp = arr[pos:pos+num_iter_in]
		temp = temp.tolist()
		arr_sub = arr_sub + temp
		pos += num_iter_in*K
	print len(arr_sub)
	plt.plot(arr_sub[:], 'r')
	"""







	"""
	## TODO: manually specify the outside iter here
	num_iter_out = 100
	num_iter_in = 100
	num_tissue = 13
	count = 0
	for iter1 in range(num_iter_out):
		for k in range(num_tissue):
			x = np.arange(count, count+num_iter_in)
			plt.plot(x, arr[x], '-', color=list_tissue_color[k])
			count += num_iter_in
	## the legend
	list_handle = []
	for k in range(num_tissue):
		line = mlines.Line2D([], [], color=list_tissue_color[k], label=list_tissue[k])
		list_handle.append(line)
	plt.legend(handles=list_handle)
	"""






