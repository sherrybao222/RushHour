# state level: mean human length, optimal length, dynamic MAG
# visualize bar plots and scatter plots for correlation and p
# save features data files
import json, math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG, solution
from scipy import stats




def get_solution(insdatafile, solutionfile):
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	board_str = MAG.board_to_str(my_board)
	print(board_str)
	sol_list = solution.main(board_str)
	print(sol_list)
	np.save(solutionfile, np.array(sol_list))

def prop_unsafe(insdatafile, datafile):
	instance_data = []
	trial_data = []
	total_unsafe_prop = 0
	is_win = False
	# filter successful trials
	with open(datafile) as f:
		for line in f:
			instance_data.append(json.loads(line))
	for i in range(0, len(instance_data)): 
		line = instance_data[i]
		win = line['meta_move']
		if win == 'win':
			is_win = True
		move_num = line['move_number']
		cur_move = line['move']
		car_to_move = cur_move[0]
		try:
   			move_to_position = int(cur_move[2:])
		except ValueError:
			print("Failure w/ value " + cur_move[2:] + ', ' + datafile + ' line' + str(i))
			continue
		if move_num == '0':
			if not is_win and len(trial_data)>0:
				trial_data.pop()
			trial_data.append([])
			trial_data[-1].append((car_to_move, move_to_position))
			is_win = False
			continue
		trial_data[-1].append((car_to_move, move_to_position))
		if i == len(instance_data) - 1:
			if not is_win:
				trial_data.pop()
	# process successful trials
	for i in range(len(trial_data)):
		unsafe = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial mag
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		n_node, n_edge = MAG.get_mag_attr(new_car_list)
		for j in range(len(trial_data[i])):
			my_board, my_red = MAG.board_move(new_car_list, \
										trial_data[i][j][0], trial_data[i][j][1])
			new_car_list = MAG.construct_mag(my_board, my_red)
			cur_node, cur_edge = MAG.get_mag_attr(new_car_list)
			if cur_edge > n_edge:
				unsafe += 1
			n_node = cur_node
			n_edge = cur_edge
		total_unsafe_prop += float(unsafe) / float(len(trial_data[i]) - 1)
		# print(float(unsafe) / float(len(trial_data[i])))

	return total_unsafe_prop / len(trial_data)


def prop_back_move(insdatafile, datafile):
	instance_data = []
	trial_data = []
	total_back_move = 0
	is_win = False
	# filter successful trials
	with open(datafile) as f:
		for line in f:
			instance_data.append(json.loads(line))
	for i in range(0, len(instance_data)): 
		line = instance_data[i]
		win = line['meta_move']
		if win == 'win':
			is_win = True
		move_num = line['move_number']
		cur_move = line['move']
		car_to_move = cur_move[0]
		try:
   			move_to_position = int(cur_move[2:])
		except ValueError:
			print("Failure w/ value " + cur_move[2:] + ', ' + datafile + ' line' + str(i))
			continue
		if move_num == '0':
			if not is_win and len(trial_data)>0:
				trial_data.pop()
			trial_data.append([])
			trial_data[-1].append((car_to_move, move_to_position))
			is_win = False
			continue
		trial_data[-1].append((car_to_move, move_to_position))
		if i == len(instance_data) - 1:
			if not is_win:
				trial_data.pop()
	# process successful trials
	for i in range(len(trial_data)):
		back = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial mag
		for j in range(len(trial_data[i])):
			my_car_list, new_red = MAG.move(my_car_list, \
											trial_data[i][j][0], trial_data[i][j][1])
			if new_red.start[0] < my_red.start[0]:
				back += 1
			my_red = new_red
		total_back_move += float(back) / float(len(trial_data[i]) - 1)
		# print(float(back) / float(len(trial_data[i])))
	# print()
	return total_back_move / len(trial_data)


def avg_node_edge(insdatafile, datafile): # average number of node and edge
	instance_data = []
	trial_data = []
	total_back_move = 0
	is_win = False
	# filter successful trials
	with open(datafile) as f:
		for line in f:
			instance_data.append(json.loads(line))
	for i in range(0, len(instance_data)): 
		line = instance_data[i]
		win = line['meta_move']
		if win == 'win':
			is_win = True
		move_num = line['move_number']
		cur_move = line['move']
		car_to_move = cur_move[0]
		try:
   			move_to_position = int(cur_move[2:])
		except ValueError:
			print("Failure w/ value " + cur_move[2:] + ', ' + datafile + ' line' + str(i))
			continue
		if move_num == '0':
			if not is_win and len(trial_data)>0:
				trial_data.pop()
			trial_data.append([])
			trial_data[-1].append((car_to_move, move_to_position))
			is_win = False
			continue
		trial_data[-1].append((car_to_move, move_to_position))
		if i == len(instance_data) - 1:
			if not is_win:
				trial_data.pop()
	# process successful trials
	useless_trial = 0
	for i in range(len(trial_data)):
		trial_node = 0
		trial_edge = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial mag
		duplicate = -1
		for j in range(len(trial_data[i])):
			my_car_list, my_red = MAG.move(my_car_list, \
										trial_data[i][j][0], trial_data[i][j][1])
			my_board, my_red = MAG.construct_board(my_car_list)
			new_car_list = MAG.construct_mag(my_board, my_red)
			n_node, n_edge = MAG.get_mag_attr(new_car_list)
			if duplicate > 3:
				print(trial_data[i])
				# break
			if n_node == 1 and n_edge == 0:
				duplicate += 1
			trial_node += n_node
			trial_edge += n_edge
		if duplicate < 0:
			duplicate = 0
		if duplicate > 3:				
			print(insdatafile)
			print()
			useless_trial += 1
			# continue
		trial_node += float(trial_node) / float(len(trial_data[i]))
		trial_edge += float(trial_edge) / float(len(trial_data[i]))

	# return float(trial_node) / float(len(trial_data) - useless_trial), \
	# 		float(trial_edge) / float(len(trial_data) - useless_trial)
	return float(trial_node) / float(len(trial_data)), \
			float(trial_edge) / float(len(trial_data))



##################################### SOLUTION WISE ############################

def prop_unsafe_solution(insdatafile, solutionfile):
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
	# return average number of cycles and average max cycle size in solution
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
	# return average number of nodes in cycles
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
	# return average number of nodes in cycles
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


# main
len_file = '/Users/chloe/Documents/RushHour/exp_data/paths.json'
move_dir = '/Users/chloe/Documents/RushHour/exp_data/'
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
data_out = '/Users/chloe/Documents/RushHour/exp_data/dynamic_MAG/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
bar_out_dir = '/Users/chloe/Documents/RushHour/state_figures/len_dynamicMAG.png'
scatter_out = '/Users/chloe/Documents/RushHour/state_figures/len_dynamicMAG_scatter.png'
scatter_x = 4
scatter_y = 4
num_features = 8
# label_list = ['human_len', 'opt_len', 'p_unsafe', 'p_backmove', 'mean_#node', 'mean_#edge', 'p_unsafe_sol', 'p_backmove_sol']
# feature_list = ['y_human', 'y_opt', 'y_unsafe', 'y_backmove', 'y_avgnode', 'y_avgedge', 'y_unsafesol' 'y_backmovesol']
label_list = ['human_len', 'opt_len', \
			'p_unsafe_sol', 'p_backmove_sol', \
			'avg_node_sol', 'avg_edge_sol', \
			'avg_ncycle_sol', 'avg_maxcycle',\
			'avg_node_incycle', 'avg_depth']
feature_list = ['y_human', 'y_opt', \
				'y_unsafesol', 'y_backmovesol', \
				'y_avgnodesol', 'y_avgedgesol', \
				'y_avgncycle', 'y_avgmaxcycle', \
				'y_avgcnode', 'y_avgdepth']
color_list = ['firebrick', 'lightcoral', 'maroon', 'salmon']
dict_list = [{} for _ in range(num_features)]
y_list = [[] for _ in range(num_features)]
y_human = []
y_opt = []
data = []
humanlen_dict = {} # human_len
mean_dict = {} # mean human_len
optimal_dict = {} # opt_len
all_human_len = [] # all human length for every puzzle
y_human_err = [] # error bar

# preprocess human & optimal len dict
for i in range(len(all_instances)):
	humanlen_dict[all_instances[i] + '_count'] = 0
	humanlen_dict[all_instances[i]+ '_len'] = 0
	all_human_len.append([])
	mean_dict[all_instances[i]] = 0
	optimal_dict[all_instances[i]] = 0
with open(len_file) as f:
	for line in f:
		data.append(json.loads(line))
for i in range(0, len(data)): # iterate through every subject trial
	line = data[i]
	instance = line['instance']
	complete = line['complete']
	human_len = int(line['human_length'])
	opt_len = int(line['optimal_length'])
	if complete == 'False':
		continue
	else:
		humanlen_dict[instance + '_count'] += 1
		humanlen_dict[instance + '_len'] = humanlen_dict[instance + '_len'] + human_len
		ins_index = all_instances.index(instance)
		all_human_len[ins_index].append(human_len)
		optimal_dict[instance] = opt_len
for i in range(len(all_instances)): # calculate mean human len and std
	if humanlen_dict[all_instances[i] + '_count'] == 0:
		mean_dict[all_instances[i]] = 0
		continue
	else:
		mean_dict[all_instances[i]] = humanlen_dict[all_instances[i] + '_len'] / humanlen_dict[all_instances[i] + '_count']
	y_human_err.append(np.std(all_human_len[i]) / math.sqrt(humanlen_dict[all_instances[i]+ '_count']))


# process mag attributes
for i in range(len(all_instances)):
	# construct mag
	cur_ins = all_instances[i]
	ins_file = ins_dir + all_instances[i] + '.json'
	move_file = move_dir + all_instances[i] + '_moves.json'
	sol_file = move_dir + all_instances[i] + '_solution.npy'
	# get_solution(ins_file, sol_file)
	# dict_list[0][cur_ins] = prop_unsafe(ins_file, move_file)
	# dict_list[1][cur_ins] = prop_back_move(ins_file, move_file)
	# dict_list[2][cur_ins], dict_list[3][cur_ins] = avg_node_edge(ins_file, move_file)
	print(cur_ins)
	dict_list[0][cur_ins] = prop_unsafe_solution(ins_file, sol_file)
	print('prop unsafe solution: ', dict_list[0][cur_ins])
	dict_list[1][cur_ins] = prop_back_move_solution(ins_file, sol_file)
	print('prop backmove solution: ', dict_list[1][cur_ins])
	dict_list[2][cur_ins], dict_list[3][cur_ins] = avg_node_edge_solution(ins_file, sol_file)
	print('avg node, edge solution: ' + str(dict_list[2][cur_ins]) \
			+ ', ' + str(dict_list[3][cur_ins]))
	dict_list[4][cur_ins], dict_list[5][cur_ins] = avg_cycle_solution(ins_file, sol_file)
	print('avg ncycle, maxcycle solution: ' + str(dict_list[4][cur_ins]) \
			+ ', ' + str(dict_list[5][cur_ins]))
	dict_list[6][cur_ins] = avg_node_cycle_solution(ins_file, sol_file)
	dict_list[7][cur_ins] = avg_red_depth_solution(ins_file, sol_file)


# generate value lists
for i in range(len(all_instances)):
	y_human.append(mean_dict[all_instances[i]])
	y_opt.append(optimal_dict[all_instances[i]])
	for j in range(0, num_features):
		y_list[j].append(dict_list[j][all_instances[i]])
		# print(len(y_list[j]))
# save data
np.save(data_out + 'y_human.npy',y_human) # mean human len
np.save(data_out + 'y_opt.npy', y_opt) # opt len
for i in range(2,len(feature_list)):
	np.save(data_out + feature_list[i] + '.npy', y_list[i-2])


# bar plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.bar(np.arange(len(all_instances)), y_human, alpha=0.9, color='orange', label='human')
# ax.errorbar(np.arange(len(all_instances)), y_human, yerr=y_human_err, alpha=0.5, c='red', fmt='none')
# ax.bar(np.arange(len(all_instances)), y_opt, alpha=0.65, color='green', label='optimal')
# ax.set_xticklabels([])
# ax.yaxis.set_major_locator(MaxNLocator(20))
# ax.grid(axis = 'y', alpha = 0.3)
# ax.set_facecolor('0.98')
# ax.set_xlabel('Puzzles')
# for i in range(len(y_list)):
# 	plt.plot(np.arange(len(all_instances)), y_list[i], color=color_list[i], label=label_list[i+2])
# plt.title('Human Length, Optimal Length, dynamic MAG')
# plt.legend(loc='upper left')
# # plt.show()
# plt.savefig(bar_out_dir)
# plt.close()


# calculate pearson correlation and p-value for human_len and MAG info
corr_list = []
p_list = []
for i in range(2, len(label_list)):
	corr, p = stats.spearmanr(y_human, y_list[i-2])
	corr_list.append(corr)
	p_list.append(p)
	print(('SP-corr human_len & ' + label_list[i] + ': %s, P-value is %s\n') % (str(format(corr, '.2g')), str(format(p, '.2g'))))


# create scatter plot to show correlation and p
fig, axarr = plt.subplots(scatter_x, scatter_y, figsize=(scatter_x*9, scatter_y*8))
count = 0
for i in range(scatter_x):
	for j in range(scatter_y):
		if count >= len(y_list):
			axarr[i,j].axis('off')
			continue
		axarr[i,j].scatter(y_human, y_list[count])
		axarr[i,j].set_ylabel(label_list[count+2], fontsize=20, fontweight='bold')
		t = 'rho=%s, p=%s'%(format(corr_list[count], '.3f'), format(p_list[count], '.2g'))
		axarr[i,j].set_title(t,y=0.85,fontsize=17)
		axarr[i,j].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
		axarr[i,j].tick_params(axis='both', labelsize=17)
		axarr[i,j].grid(True, axis='x', alpha=0.3)
		count += 1

plt.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5, rect=None) 
plt.suptitle('human_len vs dynamic MAG', y=0.999, fontsize=22, fontweight='bold')
# plt.show()
plt.savefig(scatter_out)
plt.close()



