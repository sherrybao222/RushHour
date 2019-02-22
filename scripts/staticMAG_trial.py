# get static attributes of the new MAG for subject trials
# save features data files
import json, math
import numpy as np
import MAG
from scipy import stats

data_out = '/Users/chloe/Documents/RushHour/state_model/in_data_trials/'
puzzle_dir = '/Users/chloe/Documents/RushHour/state_model/in_data_trials/puzzle_list.npy'
ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'


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


# load puzzle list
all_puzzles = np.load(puzzle_dir)

# process mag attributes
for i in range(len(all_puzzles)):
	# construct mag
	cur_ins = all_puzzles[i]
	ins_dir = ins_file + all_puzzles[i] + '.json'
	my_car_list, my_red = MAG.json_to_car_list(ins_dir)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)

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
	# cycle information
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


# save data
np.save(data_out + 'y_nodes.npy', y_nodes) # num of nodes
np.save(data_out + 'y_edges.npy', y_edges) # num of edges
np.save(data_out + 'y_en.npy', y_en) # num edge/num node
np.save(data_out + 'y_enp.npy', y_enp) # num edge/(num node - num leaf)
np.save(data_out + 'y_e2n.npy', y_e2n) # num edge sq / num node
np.save(data_out + 'y_countscc.npy', y_countscc) # num of scc
np.save(data_out + 'y_maxscc.npy', y_maxscc) # max scc size
np.save(data_out + 'y_countcycle.npy', y_countcycle) # num of cycles
np.save(data_out + 'y_maxcycle.npy', y_maxcycle) # max cycle size
np.save(data_out + 'y_c_incycle.npy', y_c_incycle) # num of cycles in cycles
np.save(data_out + 'y_nnc.npy', y_nnc) # num of nodes in cycles
np.save(data_out + 'y_pnc.npy', y_pnc) # proportion of nodes in cycles
np.save(data_out + 'y_depth.npy', y_depth) # len of longest path from red
np.save(data_out + 'y_ndepth.npy', y_ndepth) # num of longest paths from red
np.save(data_out + 'y_gcluster.npy', y_gcluster) # global clustering coefficient
np.save(data_out + 'y_lcluster.npy', y_lcluster) # mean local clustering coefficient
