# prepare data for state level analysis
# dynamic MAG features from human moves (for each subject trial)
# visualize all MAG features
import json, math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG, solution
from scipy import stats

# main
len_file = '/Users/chloe/Documents/RushHour/exp_data/paths.json'
move_dir = '/Users/chloe/Documents/RushHour/exp_data/'
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
data_dir = '/Users/chloe/Documents/RushHour/state_model/in_data_trials/'
valid_sub_dir = '/Users/chloe/Documents/RushHour/exp_data/valid_sub.npy'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
label_list = ['human_len', 'opt_len', \
			'p_unsafe_sol', 'p_backmove_sol', \
			'avg_node_sol', 'avg_edge_sol', \
			'avg_ncycle_sol', 'avg_maxcycle_sol',\
			'avg_node_incycle_sol', 'avg_depth_sol',\
			'node_rate', 'edge_rate']
feature_list = ['y_human', 'y_opt', \
				'y_unsafeSol', 'y_backMoveSol', \
				'y_avgNodeSol', 'y_avgEdgeSol', \
				'y_avgnCycleSol', 'y_avgMaxCycleSol', \
				'y_avgcNodeSol', 'y_avgDepthSol',\
				'y_nodeRate', 'y_edgeRate']
num_features = len(feature_list) - 2
# valid includes bonus and postquestionare
valid_subjects = np.load(valid_sub_dir)


##################################### SOLUTION WISE #################################

def prop_unsafe_solution(insdatafile, solutionfile):
	# proportion of unsafe moves in solution file
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	sol_list = np.load(solutionfile)
	unsafe = 0
	for move in sol_list:
		car_to_move = move[0]
		if car_to_move == 'R':
			car_to_move = 'r'
		move_by = int(move[2:])
		my_car_list, my_red = MAG.move_by(my_car_list, \
										car_to_move, move_by)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		cur_node, cur_edge = MAG.get_mag_attr(new_car_list)
		if cur_edge > n_edge:
			unsafe += 1
		n_node = cur_node
		n_edge = cur_edge
	return float(unsafe) / float(len(sol_list) + 1)


def prop_back_move_solution(insdatafile, solutionfile):
	# proportion of red-backward moves in the solution file
	sol_list = np.load(solutionfile)
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	back = 0
	for move in sol_list:
		car_to_move = move[0]
		if car_to_move == 'R':
			car_to_move = 'r'
		move_by = int(move[2:])
		my_car_list, new_red = MAG.move_by(my_car_list, \
										car_to_move, move_by)
		if new_red.start[0] < my_red.start[0]:
				back += 1
		my_red = new_red
	return float(back) / float(len(sol_list) + 1)


def avg_node_edge_solution(insdatafile, solutionfile):
	# average number of nodes and edges in solution file
	sol_list = np.load(solutionfile)
	total_node = 0
	total_edge = 0
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	total_node += n_node
	total_edge += n_edge
	for move in sol_list:
		car_to_move = move[0]
		if car_to_move == 'R':
			car_to_move = 'r'
		move_by = int(move[2:])
		my_car_list, my_red = MAG.move_by(my_car_list, \
										car_to_move, move_by)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		n_node, n_edge = MAG.get_mag_attr(new_car_list)
		total_node += n_node
		total_edge += n_edge
	return float(total_node) / float(len(sol_list) + 1),\
			float(total_edge) / float(len(sol_list) + 1)


def avg_cycle_solution(insdatafile, solutionfile):
	# average number of cycles and average max cycle size in solution file
	sol_list = np.load(solutionfile)
	total_ncycle = 0
	total_maxcycle = 0
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	ncycle, _, maxcycle = MAG.find_cycles(new_car_list) 
	total_ncycle += ncycle
	total_maxcycle += maxcycle
	for move in sol_list:
		car_to_move = move[0]
		if car_to_move == 'R':
			car_to_move = 'r'
		move_by = int(move[2:])
		my_board, new_red = MAG.move_by(my_car_list, \
									car_to_move, move_by)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		ncycle, _, maxcycle = MAG.find_cycles(new_car_list) 
		total_ncycle += ncycle
		total_maxcycle += maxcycle
	return float(total_ncycle) / float(len(sol_list) + 1),\
			float(total_maxcycle) / float(len(sol_list) + 1)


def avg_node_cycle_solution(insdatafile, solutionfile):
	# average number of nodes in cycles in solution file
	sol_list = np.load(solutionfile)
	total_cnode = 0
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	cnode, _ = MAG.num_nodes_in_cycle(new_car_list) 
	total_cnode += cnode
	for move in sol_list:
		car_to_move = move[0]
		if car_to_move == 'R':
			car_to_move = 'r'
		move_by = int(move[2:])
		my_board, new_red = MAG.move_by(my_car_list, \
									car_to_move, move_by)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		cnode, _ = MAG.num_nodes_in_cycle(new_car_list) 
		total_cnode += cnode
	return float(total_cnode) / float(len(sol_list) + 1)


def avg_red_depth_solution(insdatafile, solutionfile):
	# average length of the longest path from red in solution file
	sol_list = np.load(solutionfile)
	total_depth = 0
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	depth, _ = MAG.longest_path(new_car_list)
	total_depth += depth
	for move in sol_list:
		car_to_move = move[0]
		if car_to_move == 'R':
			car_to_move = 'r'
		move_by = int(move[2:])
		my_board, new_red = MAG.move_by(my_car_list, \
									car_to_move, move_by)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		depth, _ = MAG.longest_path(new_car_list)
		total_depth += depth
	return float(total_depth) / float(len(sol_list) + 1)


######################################### BOTH #######################################

def node_edge_rate(insdatafile, opt_len):
	# initial number of nodes divided by opt_len, initial number of edges divided by opt_len
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	return float(n_node) / float(opt_len), float(n_edge) / float(opt_len)


################################## MAIN PROGRAM BEGINS ################################
data_dir = '/Users/chloe/Documents/RushHour/state_model/in_data_trials/'
sol_dir = '/Users/chloe/Documents/RushHour/exp_data/'
puzzle_list = np.load(data_dir + 'puzzle_list.npy')
optlen_list = np.load(data_dir + 'optlen_list.npy')
y_list = [[] for _ in range(num_features)]

# process mag attributes
for i in range(len(puzzle_list)):
	# construct mag
	cur_ins = puzzle_list[i]
	ins_file = ins_dir + cur_ins + '.json'
	sol_file = move_dir + cur_ins + '_solution.npy'	
	# MAG features from solution file
	y_list[0].append(prop_unsafe_solution(ins_file, sol_file))
	# print('prop unsafe solution: ', dict_list[0][cur_ins])
	y_list[1].append(prop_back_move_solution(ins_file, sol_file))
	# print('prop backmove solution: ', dict_list[1][cur_ins])
	one, two = avg_node_edge_solution(ins_file, sol_file)
	y_list[2].append(one)
	y_list[3].append(two)
	# print('avg node, edge solution: ' + str(dict_list[2][cur_ins]) \
			# + ', ' + str(dict_list[3][cur_ins]))
	one, two = avg_cycle_solution(ins_file, sol_file)
	y_list[4].append(one)
	y_list[5].append(two)
	# print('avg ncycle, maxcycle solution: ' + str(dict_list[4][cur_ins]) \
			# + ', ' + str(dict_list[5][cur_ins]))
	y_list[6].append(avg_node_cycle_solution(ins_file, sol_file))
	y_list[7].append(avg_red_depth_solution(ins_file, sol_file))
	one, two = node_edge_rate(ins_file, optlen_list[i])
	y_list[8].append(one)
	y_list[9].append(two)
	# print('node rate, edge rate:' + str(dict_list[8][cur_ins]) + ' ' + str(dict_list[9][cur_ins]))

for i in range(len(y_list)):
	print(len(y_list[i]))
	print(y_list[i])

# save data
for i in range(2, len(feature_list)):
	np.save(data_dir + feature_list[i] + '.npy', y_list[i-2])



stop

# # calculate pearson correlation and p-value for human_len and MAG info
# corr_list = []
# p_list = []
# for i in range(2, len(label_list)):
# 	corr, p = stats.spearmanr(y_human, y_list[i-2])
# 	corr_list.append(corr)
# 	p_list.append(p)
# 	print(('SP-corr human_len & ' + label_list[i] + ': %s, P-value is %s\n') % (str(format(corr, '.2g')), str(format(p, '.2g'))))


# # create scatter plot to show correlation and p
# fig, axarr = plt.subplots(scatter_x, scatter_y, figsize=(scatter_x*9, scatter_y*8))
# count = 0
# for i in range(scatter_x):
# 	for j in range(scatter_y):
# 		if count >= len(y_list):
# 			axarr[i,j].axis('off')
# 			continue
# 		axarr[i,j].scatter(y_list[count], y_human)
# 		axarr[i,j].set_xlabel(label_list[count+2], fontsize=20, fontweight='bold')
# 		t = 'rho=%s, p=%s'%(format(corr_list[count], '.3f'), format(p_list[count], '.2g'))
# 		axarr[i,j].set_title(t,y=0.85,fontsize=17)
# 		axarr[i,j].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
# 		axarr[i,j].xaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
# 		axarr[i,j].tick_params(axis='both', labelsize=17)
# 		axarr[i,j].grid(True, axis='y', alpha=0.3)
# 		count += 1

# plt.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5, rect=None) 
# plt.suptitle('human_len vs dynamic MAG', y=0.999, fontsize=22, fontweight='bold')
# # plt.show()
# plt.savefig(scatter_out)
# plt.close()

effect_avgDepthSol
effect_avgEdgeSol
effect_avgMaxCycleSol
effect_avgNodeSol
effect_avgcNodeSol
effect_avgnCycleSol
effect_backMoveSol
effect_edgeRate
effect_nodeRate
effect_unsafeSol
intercept

#       effect_avgDepthSol_val,
#       effect_avgEdgeSol_val,
#       effect_avgMaxCycleSol_val,
#       effect_avgNodeSol_val,
#       effect_avgcNodeSol_val,
#       effect_avgnCycleSol_val,
#       effect_backMoveSol_val,
#       effect_edgeRate_val,
#       effect_nodeRate_val,
#       effect_unsafeSol_val,

