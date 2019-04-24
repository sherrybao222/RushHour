# generate MAG features for each move
# static MAG features for current state
# no optimal solution used
# need to run with py365
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time


trialdata = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
out_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed_static_MAG.csv'
label_list = ['node', 'edge', \
			'e/n', 'e/np', \
			'e2n', 'scc',\
			'maxscc', 'cycle',\
			'maxcycle', 'cincycle',\
			'ninc', 'pnc',\
			'depth', 'ndepth',\
			'gcluster', 'lcluster',\
			'mobility']

# initialize attribute lists
y_nodes = [] # number of nodes
y_edges = [] # number of edges
y_en = [] # number of edges / number of nodes
y_enp = [] # number of edges / (number of nodes - number of leaf)
y_e2n = [] # number of edges ^ 2 / number of nodes
y_countscc = [] # number of SCC
y_maxscc = [] # max SCC size
y_countcycle = [] # number of cycles
y_maxcycle = [] # max cycle size
y_c_incycle = [] # number of cycles in cycles
y_nnc = [] # number of nodes in cycles
y_pnc = [] # proportion of nodes in cycles
y_depth = [] # the longest path length from red
y_ndepth = [] # number of the longest paths from red
y_gcluster = [] # global clustering coefficient
y_lcluster = [] # local or the mean average clustering coefficient
y_mobility = []

move_data = pd.read_csv(trialdata)
my_car_list, my_red, my_board, new_car_list = '', '', '', ''

# process each data row
for i in range(len(move_data)):
	row = move_data.loc[i, :]
	event = row['event']
	# construct new board if start new game
	if event == 'start':
		puzzle = row['instance']
		insdatafile = instance_folder + puzzle + '.json'
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial mag
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
	else: # make move
		piece = row['piece']
		moveto = int(row['move'])
		my_board, my_red = MAG.board_move(new_car_list, piece, moveto)
		new_car_list = MAG.construct_mag(my_board, my_red)

	print(i+3)
	# num nodes, num edges
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	y_nodes.append(n_node)
	y_edges.append(n_edge)
	# edge nodes ratio, branching factor
	ebn = MAG.e_by_n(new_car_list)
	ebpn = MAG.e_by_pn(new_car_list)
	e2n = MAG.e2_by_n(new_car_list)
	y_en.append(ebn)
	y_enp.append(ebpn)
	y_e2n.append(e2n)
	# SCC factors
	countscc, _, maxlen = MAG.find_SCC(new_car_list)
	y_countscc.append(countscc)
	y_maxscc.append(maxlen)
	# cycle count, max cycle size
	countc, _, maxc = MAG.find_cycles(new_car_list) 
	y_countcycle.append(countc)
	y_maxcycle.append(maxc)
	# number of cycles in cycles
	c_cycle = MAG.num_in_cycles(new_car_list)
	y_c_incycle.append(c_cycle)
	# number of nodes in cycles
	n_nc, _ = MAG.num_nodes_in_cycle(new_car_list)
	y_nnc.append(n_nc)
	# proportion of nodes in cycles
	pro = MAG.pro_nodes_in_cycle(new_car_list)
	y_pnc.append(pro)
	# longest path len from red and number of longest paths
	depth, paths = MAG.longest_path(new_car_list)
	y_depth.append(depth)
	y_ndepth.append(len(paths))
	# clustering coefficient
	gcluster = MAG.global_cluster_coef(new_car_list)
	lcluster = MAG.av_local_cluster_coef(new_car_list)
	y_gcluster.append(gcluster)
	y_lcluster.append(lcluster)
	# mobility
	mobility = MAG.board_freedom(my_board)
	y_mobility.append(mobility)


with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('node_human_static', 'edge_human_static', \
			'en_human_static', 'enp_human_static', \
			'e2n_human_static', 'scc_human_static',\
			'maxscc_human_static', 'cycle_human_static',\
			'maxcycle_human_static', 'cincycle_human_static',\
			'ninc_human_static', 'pnc_human_static',\
			'depth_human_static', 'ndepth_human_static',\
			'gcluster_human_static', 'lcluster_human_static',\
			'mobility'))
	for j in range(len(y_nodes)):
		writer.writerow([y_nodes[j], y_edges[j], \
						y_en[j], y_enp[j], \
						y_e2n[j], y_countscc[j], \
						y_maxscc[j], y_countcycle[j],\
						y_maxcycle[j], y_c_incycle[j],\
						y_nnc[j], y_pnc[j], \
						y_depth[j], y_ndepth[j],\
						y_gcluster[j], y_lcluster[j],\
						y_mobility[j]])






