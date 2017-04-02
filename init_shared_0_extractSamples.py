## tasks:
##	1. pull out of the data, given: tissue list, gene list, individual list (to make things comparable with GTMLNN)
##	2. normalize using Z score
## 
## dir: "~/GTEx_gtmtm/preprocess/"



##=====================
##==== libraries
##=====================
import math
import sys
import time
import os
import numpy as np
from scipy import stats
import re








##===============
##==== sub-routines
##===============
# get the "xxx-yyy" from "xxx-yyy-zzz-aaa-qqq", which is defined as the individual ID of the GTEx samples
pattern_indiv = re.compile(r'^(\w)+([\-])(\w)+')
def get_individual_id(s):
	match = pattern_indiv.match(s)
	if match:
		return match.group()
	else:
		print "!!! no individual ID is found..."
		return ""







if __name__ == '__main__':






	##======================================================================================================
	##==== pick up samples that's qualified
	##======================================================================================================
	## sample_tissue_rep
	#file = open("./data_raw/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_sample_tissue_type", 'r')
	file = open("../../GTEx_data/data_raw/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_sample_tissue_type", 'r')
	sample_tissue_rep = {}
	while 1:
		line = file.readline()[:-1]
		if not line:
			break

		line = line.split('\t')

		if len(line) < 3:
			print "we have no info for:",
			print line
			continue

		sample = line[0]
		tissue = line[2]

		sample_tissue_rep[sample] = tissue
	file.close()



	##============ process the rpkm matrix to get eQTL samples ==============
	## get the sample_rep first
	sample_rep = {}
	#file = open("./data_raw/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_tissue_sample_rep", 'r')
	file = open("../../GTEx_data/data_raw/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_tissue_sample_rep", 'r')
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split('\t')[1:]
		for sample in line:
			sample_rep[sample] = 1
	file.close()




	# filter all the samples again
	#file = open("./data_raw/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_rpkm.gct_1_genotype", 'r')
	file = open("../../GTEx_data/data_raw/GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_rpkm.gct_1_genotype", 'r')
	list_sample = []
	index_rep = {}
	line = (file.readline()).strip()
	line = (line.split('\t'))[1:]
	for i in range(len(line)):
		sample = line[i]
		if sample in sample_rep:
			index_rep[i] = 1
			list_sample.append(sample)
	list_sample = np.array(list_sample)
	np.save("./data_raw/list_sample", list_sample)
	print "there are # of samples totally:",
	print len(list_sample)
	
	Data = []
	list_gene = []
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split('\t')
		gene = line[0]
		list_gene.append(gene)
		rpkm_list = map(lambda x: float(x), line[1:])
		Data.append([])
		for i in range(len(rpkm_list)):
			if i in index_rep:
				rpkm = rpkm_list[i]
				Data[-1].append(rpkm)
	file.close()
	Data = np.array(Data)
	np.save("./data_raw/Data", Data)
	print Data.shape







	##======================================================================================================
	##==== pick up genes from specified list
	##======================================================================================================
	## we have:
	#Data = []
	#list_gene = []
	print "now we are picking up a subset of all the genes..."
	Gene_list = np.load("/ifs/scratch/c2b2/ip_lab/sy2515/GTEx_gtmnn/preprocess/data_prepared/Gene_list.npy")
	repo_Gene_list = {}
	for gene in Gene_list:
		repo_Gene_list[gene] = 1
	print "there are # of qualified genes:", len(repo_Gene_list)

	Data_null = []
	list_gene_final = []
	for i in range(len(Data)):
		gene = list_gene[i]
		rpkm_list = Data[i]
		if gene not in repo_Gene_list:
			continue
		list_gene_final.append(gene)
		Data_null.append(rpkm_list)
	list_gene = np.array(list_gene_final)
	## NOTE
	print "# of genes:",
	print len(list_gene)
	Data_null = np.array(Data_null)
	print "shape of the data matrix (before norm):",
	print Data_null.shape






	##=============================================================================================
	##==== normalizing all the samples
	##=============================================================================================
	## we have:
	#Data_null = []
	## the following several normalization methods:
	Data_norm = []
	print "z normalization..."
	for i in range(len(Data_null)):
		rpkm_list = Data_null[i]
		rpkm_list = stats.zscore(rpkm_list)
		Data_norm.append(rpkm_list)
	Data_norm = np.array(Data_norm)
	print "Data_norm shape:", Data_norm.shape






	# ##=============================================================================================
	# ##==== normalizing all the samples
	# ##=============================================================================================
	## sample_tissue_rep
	## list_sample
	list_tissue = np.load("../../GTEx_gtmnn/preprocess/data_prepared/Tissue_list.npy")
	list_individual = np.load("../../GTEx_gtmnn/preprocess/data_prepared/Individual_list.npy")
	list_gene = np.load("../../GTEx_gtmnn/preprocess/data_prepared/Gene_list.npy")
	K = len(list_tissue)
	I = len(list_individual)
	J = len(list_gene)
	Data_final = np.zeros((K, I, J)) + float("Nan")

	#
	repo_index_tissue = {}
	for k in range(len(list_tissue)):
		tissue = list_tissue[k]
		repo_index_tissue[tissue] = k

	#
	repo_index_individual = {}
	for i in range(len(list_individual)):
		individual = list_individual[i]
		repo_index_individual[individual] = i

	#
	Data_norm = Data_norm.T
	for index in range(len(Data_norm)):
		sample = list_sample[index]
		rpkm_list = Data_norm[index]
		#
		tissue = sample_tissue_rep[sample]
		index_tissue = repo_index_tissue[tissue]
		#
		individual = get_individual_id(sample)
		index_individual = repo_index_individual[individual]

		Data_final[index_tissue][index_individual] = rpkm_list
	np.save("./data_raw/Data_final", Data_final)















