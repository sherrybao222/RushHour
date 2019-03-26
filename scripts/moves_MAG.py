# generate MAG features for each optimal move
# need to run with py365
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time


moves_file = '/Users/chloe/Documents/RushHour/exp_data/moves_valid.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
move_sol_dir = '/Users/chloe/Documents/RushHour/exp_data/moves_sol1000_final.npy'
out_file = '/Users/chloe/Documents/RushHour/exp_data/moves_MAG1000.csv'
out_file1 = '/Users/chloe/Documents/RushHour/exp_data/moves_MAG1000.npy'
label_list = ['p_unsafe_sol', 'p_backmove_sol', \
			'avg_node_sol', 'avg_edge_sol', \
			'avg_ncycle_sol', 'avg_maxcycle_sol',\
			'avg_node_incycle_sol', 'avg_depth_sol',\
			'node_rate', 'edge_rate']
feature_list = ['y_unsafeSol', 'y_backMoveSol', \
				'y_avgNodeSol', 'y_avgEdgeSol', \
				'y_avgnCycleSol', 'y_avgMaxCycleSol', \
				'y_avgcNodeSol', 'y_avgDepthSol',\
				'y_nodeRate', 'y_edgeRate']



##################################### SOLUTION WISE #################################

def prop_unsafe_solution(in_car_list, in_board, in_red, solution):
	# proportion of unsafe moves in solution file
	my_car_list = in_car_list.copy()
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	sol_list = solution
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


def prop_back_move_solution(in_car_list, in_board, in_red, solution):
	# proportion of red-backward moves in the solution file
	my_car_list = in_car_list.copy()
	my_board, my_red = MAG.construct_board(my_car_list)
	sol_list = solution
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


def avg_node_edge_solution(in_car_list, in_board, in_red, solution):
	# average number of nodes and edges in solution file
	sol_list = solution
	my_car_list = in_car_list.copy()
	my_board, my_red = MAG.construct_board(my_car_list)
	total_node = 0
	total_edge = 0
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


def avg_cycle_solution(in_car_list, in_board, in_red, solution):
	# average number of cycles and average max cycle size in solution file
	sol_list = solution
	my_car_list = in_car_list.copy()
	my_board, my_red = MAG.construct_board(my_car_list)
	total_ncycle = 0
	total_maxcycle = 0
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


def avg_node_cycle_solution(in_car_list, in_board, in_red, solution):
	# average number of nodes in cycles in solution file
	sol_list = solution
	my_car_list = in_car_list.copy()
	my_board, my_red = MAG.construct_board(my_car_list)
	total_cnode = 0
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


def avg_red_depth_solution(in_car_list, in_board, in_red, solution):
	# average length of the longest path from red in solution file
	sol_list = solution
	my_car_list = in_car_list.copy()
	my_board, my_red = MAG.construct_board(my_car_list)
	total_depth = 0
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

def node_edge_rate(in_car_list, in_board, in_red, opt_len):
	# initial number of nodes divided by opt_len, initial number of edges divided by opt_len
	my_car_list = in_car_list.copy()
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	return float(n_node) / float(opt_len), float(n_edge) / float(opt_len)







#################################### MAIN PROGRAM ###############################

y_list = []
move_sol = np.load(move_sol_dir)
move_data = pd.read_csv(moves_file)


for i in range(0, len(move_sol)):

	solutions = move_sol[i]

	row = move_data.loc[i, :]
	cur_subject = row['worker'] + ':' + row['assignment']
	cur_instance = row['instance']
	cur_move = row['move']
	move_number = row['move_number']
	trial_number = row['trial_number']
	meta_move = row['meta_move']
	opt_len = row['optlen']

	if meta_move == 'win' or cur_move == 'r@16' or trial_number == 0 or solutions == []:
		y_list.append([0.0] * len(feature_list))
		continue

	if move_number == 0: # initialize puzzle
		infile = instance_folder + cur_instance + '.json'
		cur_car_list, cur_red = MAG.json_to_car_list(infile)
		cur_board, cur_red = MAG.construct_board(cur_car_list)
		# optimal solutions at the beginning of each trial
		board_str = MAG.board_to_str(cur_board)
		######################### process solutions MAG here
		cur_y_all = []
		for sol in solutions:
			for m in sol:
				print(m)
			print(type(sol))
			sol = list(sol)
			this_y = []
			this_y.append(prop_unsafe_solution(cur_car_list, cur_board, cur_red, sol))
			this_y.append(prop_back_move_solution(cur_car_list, cur_board, cur_red, sol))
			one, two = avg_node_edge_solution(cur_car_list, cur_board, cur_red, sol)
			this_y.append(one)
			this_y.append(two)
			one, two = avg_cycle_solution(cur_car_list, cur_board, cur_red, sol)
			this_y.append(one)
			this_y.append(two)
			this_y.append(avg_node_cycle_solution(cur_car_list, cur_board, cur_red, sol))
			this_y.append(avg_red_depth_solution(cur_car_list, cur_board, cur_red, sol))
			one, two = node_edge_rate(cur_car_list, cur_board, cur_red, opt_len)
			this_y.append(one)
			this_y.append(two)
			cur_y_all.append(this_y)
		# take average of all solutions 
		cur_y = np.mean(np.array(cur_y_all), axis=0)
		y_list.append(cur_y)
		car_id = cur_move[0]
		car_to_pos = int(cur_move[2:])
		continue
	
	# new board
	cur_board, cur_red = MAG.construct_board(cur_car_list)
	# make move
	cur_car_list, cur_red = MAG.move(cur_car_list, car_id, car_to_pos)
	cur_board, cur_red = MAG.construct_board(cur_car_list)
	board_str = MAG.board_to_str(cur_board)
	cur_y_all = []
	for sol in solutions:
		this_y = []
		this_y.append(prop_unsafe_solution(cur_car_list, cur_board, cur_red, sol))
		this_y.append(prop_back_move_solution(cur_car_list, cur_board, cur_red, sol))
		one, two = avg_node_edge_solution(cur_car_list, cur_board, cur_red, sol)
		this_y.append(one)
		this_y.append(two)
		one, two = avg_cycle_solution(cur_car_list, cur_board, cur_red, sol)
		this_y.append(one)
		this_y.append(two)
		this_y.append(avg_node_cycle_solution(cur_car_list, cur_board, cur_red, sol))
		this_y.append(avg_red_depth_solution(cur_car_list, cur_board, cur_red, sol))
		one, two = node_edge_rate(cur_car_list, cur_board, cur_red, opt_len)
		this_y.append(one)
		this_y.append(two)
		cur_y_all.append(this_y)
	# take average of all solutions 
	cur_y = np.mean(np.array(cur_y_all), axis=0)
	y_list.append(cur_y)
	print("iteration count ", i)
	print("\n\n")
	
	# for next move
	car_id = cur_move[0]
	car_to_pos = int(cur_move[2:])

	# end of file
	if i == len(move_data) - 1:
		# new board
		cur_board, cur_red = MAG.construct_board(cur_car_list)
		# make move
		cur_car_list, cur_red = MAG.move(cur_car_list, car_id, car_to_pos)
		cur_board, cur_red = MAG.construct_board(cur_car_list)
		board_str = MAG.board_to_str(cur_board)
		cur_y_all = []
		for sol in solutions:
			this_y = []
			this_y.append(prop_unsafe_solution(cur_car_list, cur_board, cur_red, sol))
			this_y.append(prop_back_move_solution(cur_car_list, cur_board, cur_red, sol))
			one, two = avg_node_edge_solution(cur_car_list, cur_board, cur_red, sol)
			this_y.append(one)
			this_y.append(two)
			one, two = avg_cycle_solution(cur_car_list, cur_board, cur_red, sol)
			this_y.append(one)
			this_y.append(two)
			this_y.append(avg_node_cycle_solution(cur_car_list, cur_board, cur_red, sol))
			this_y.append(avg_red_depth_solution(cur_car_list, cur_board, cur_red, sol))
			one, two = node_edge_rate(cur_car_list, cur_board, cur_red, opt_len)
			this_y.append(one)
			this_y.append(two)
			cur_y_all.append(this_y)
		# take average of all solutions 
		cur_y = np.mean(np.array(cur_y_all), axis=0)
		y_list.append(cur_y)

	# print(y_list)
	# sys.exit()

# save y_list
np.save(out_file1, y_list)

with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('p_unsafe_sol', 'p_backmove_sol',\
					'avg_node_sol', 'avg_edge_sol', \
				'avg_ncycle_sol', 'avg_maxcycle_sol',\
				'avg_node_incycle_sol', 'avg_depth_sol',\
				'node_rate', 'edge_rate'))
	for j in range(len(y_list)):
		print(y_list[j])
		writer.writerow(y_list[j])
