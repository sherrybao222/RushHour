# save MAG info for each puzzle to txt files
import json, os, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG, Graph

datafile = '/Users/chloe/Documents/RushHour/data/data_adopted/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/'

# iterate through each puzzle
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	out_file = out_dir + instance + '/' + instance + '_MAG_info.txt'
	print(instance)
	out_string = ''
	my_car_list, my_red = MAG.json_to_car_list(datafile + instance + '.json')
	my_board = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	out_string += "#nodes: " + str(n_node) + ", #edges: " + str(n_edge) + '\n'
	
	countscc, scclist, maxlen = MAG.find_SCC(new_car_list)
	scclist = MAG.replace(scclist, 8, 'R')
	out_string += "#SCC: " + str(countscc) +  ", max SCC len: " +  str(maxlen) + '\n'
	out_string += 'SCC list:\n'
	for s in scclist:
		out_string += str(s) + '\n'
	
	countc, clist, maxc = MAG.find_cycles(new_car_list)
	clist = MAG.replace(clist, 8, 'R')
	out_string += "#cycles: "+ str(countc)+ ", max cycle len: "+ str(maxc) \
					+ '\n#cycles in cycle: ' + str(MAG.num_in_cycles(new_car_list)) + '\n'
	out_string += 'cycle list:\n'
	for c in clist:
		out_string += str(c) + '\n'
	
	depth, paths = MAG.longest_path(new_car_list)
	paths = MAG.replace(paths, 8, 'R')
	out_string += "longest path len from red: " + str(depth) + '\n'
	out_string += 'longest paths:\n'
	for p in paths:
		out_string += str(p) + '\n'

	n_nc, cycle_nodes = MAG.num_nodes_in_cycle(new_car_list)
	cycle_nodes = MAG.replace_1d(cycle_nodes, 8, 'R')
	pro = MAG.pro_nodes_in_cycle(new_car_list)
	out_string += '#nodes in cycles: ' + str(n_nc) \
			+ '\nproportion of nodes in cycles: ' + str(pro)\
			+ '\nlist of nodes in cycles:\n' + str(cycle_nodes)

	ebn = MAG.e_by_n(new_car_list)
	ebpn = MAG.e_by_pn(new_car_list)
	out_string += '\n#edges / #nodes = ' + str(ebn)\
			+ '\n#edges / (#nodes - #leaf) = ' + str(ebpn)
	
	with open(out_file, "w") as text_file: # save file
		text_file.write(out_string)





