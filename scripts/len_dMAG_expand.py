# state level analysis: dynamic MAG features from solution and human moves 
# visualize bar plots and scatter plots for correlation and p
# save data files, including mean human length, optimal length, dynamic MAG
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
data_out = '/Users/chloe/Documents/RushHour/state_model/in_data2/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
bar_out_dir = '/Users/chloe/Documents/RushHour/state_figures/len_dynamicMAG_exp.png'
scatter_out = '/Users/chloe/Documents/RushHour/state_figures/len_dynamicMAG_exp_scatter.png'
scatter_y = 4 # number of scatter plots alligned horizontally
scatter_x = 4 # number of scatter plots alligned vertically
num_features = 10
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
# color_list = ['firebrick', 'lightcoral', 'maroon', 'salmon']
# valid includes bonus and postquestionare
valid_subjects = ['ARWF605I7RWM7:3AZHRG4CU5SYU12YRVPCSZALW5003R', \
					'A289D98Z4GAZ28:3ZV9H2YQQEFR2R3JK2IXZUJPXAF3WW', \
					'A58QOLDFC179Q:3ON104KXQL4CKNMNKGNG9ZBVBAB4WI', \
					'A191V7PT3DQKDP:3PMBY0YE28B43VMUKKJ6EDF85RV9C8', \
					'AU4KK3OYS2UZF:3ZSY5X72NYJBGKFJ46SJ0Y9J0RLROB', \
					'A2MYB6MLQW0IGN:3TYCR1GOTDRCCQYD1V64UK7OE48LZS', \
					'A214HWAW1PYWO8:34BBWHLWHBJ6SUL255PK30LEGY5IWG', \
					'A18QU0YQB6Q8DF:3VE8AYVF8N5BS2NU6U3TMN50JMNF8V', \
					'A3Q0XAGQ7TD7MP:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB', \
					'A21S7MA9DCW95E:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A2RTJ6BDY0DZVZ:3IJXV6UZ1YR1KY4G6BFEG1DXP7SRI1', \
					'A3NMQ3019X6YE0:3I2PTA7R3U2SESF4TZBQORI5LSJQKK', \
					'A18QU0YQB6Q8DF:3VE8AYVF8N5BS2NU6U3TMN50JMNF8V', \
					'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB', \
					'A3Q0XAGQ7TD7MP:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A21S7MA9DCW95E:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A2RTJ6BDY0DZVZ:3IJXV6UZ1YR1KY4G6BFEG1DXP7SRI1', \
					'A3NMQ3019X6YE0:3I2PTA7R3U2SESF4TZBQORI5LSJQKK', \
					'A15781PHGW377Y:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I', \
					'A2IBQA01NR5N76:3X4MXAO0BHWJLTOLVSJTHSM54OHWRG', \
					'A23437BMZ5T1FH:3IHR8NYAM89M0EPM8U9LH53ZJDTP4U', \
					'A18QU0YQB6Q8DF:3VE8AYVF8N5BS2NU6U3TMN50JMNF8V', \
					'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB', \
					'A21S7MA9DCW95E:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A2RTJ6BDY0DZVZ:3IJXV6UZ1YR1KY4G6BFEG1DXP7SRI1', \
					'A3NMQ3019X6YE0:3I2PTA7R3U2SESF4TZBQORI5LSJQKK', \
					'A15781PHGW377Y:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I', \
					'A2IBQA01NR5N76:3X4MXAO0BHWJLTOLVSJTHSM54OHWRG', \
					'A23437BMZ5T1FH:3IHR8NYAM89M0EPM8U9LH53ZJDTP4U', \
					'A1LR0VQIHQUJAM:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I', \
					'A1F4N58CAX8IMK:3TS1AR6UQRM7SOIBWPBN8N9589Y7FV', \
					'A1XDMS0KFSF5JW:3M0BCWMB8W4W5M7WZVX3HDH1M3FBWL', \
					'A15FXHC1CVNW31:3TS1AR6UQRM7SOIBWPBN8N9589Y7FV', \
					'A1NT8BT94ME6F5:31LM9EDVOM0C0BWUVMJXJINN3QPJNI', \
					'A30KYQGABO7JER:37TD41K0AIHM8AITTQJXV8KY05QCSB', \
					'A3NMQ3019X6YE0:3V5Q80FXIYZ5QB5C6ITQBN30WY823U', \
					'A13BZCNJ0WR1T7:3TYCR1GOTDRCCQYD1V64UK7OHTBZLQ', \
					'A1EY7WONSYGBVY:3O6CYIULEE9B1LG2ZMEYM39PDJKUWB', \
					'A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9', \
					'A3GXC3VG37CQ3G:3TPWUS5F8A9FFRZ2DVTYSXNJ6BWWC6', \
					'A3CTXNQ2GXIQSP:34HJIJKLP64Z5YMIU6IKNXSH7PDV4I', \
					'A3BPRPN10HJD4B:3TXD01ZLD5PZSJXIPG8FRBQYTI9U4X', \
					'A23XJ8I86R0O3B:30IQTZXKALEAAZ9CBKW0ZFZP6O5X07', \
					'A3OB6REG5CGHK1:3STRJBFXOXZ5687WA35LTWTS7QZKTT', \
					'A3Q0XAGQ7TD7MP:3GLB5JMZFY3TNXFGYMKRQ0JDXNTDGZ', \
					'AXKM02NVXNGOM:33PPO7FECWN7JOLBOAKUBCWTC0AIDL', \
					'A1USR9JCAMDGM3:3PB5A5BD0WED6OE679H5Q89HBDGG71', \
					'A1F4N58CAX8IMK:35DR22AR5ES6RR89U7EJ1DXW91AX3P', \
					'A1N1EF0MIRSEZZ:3R5F3LQFV3SKIB1AENMWM1BICT5OZB', \
					'A1GQS6USF2JEYG:33F859I567LE8WC74WB3GA7E94XBHW', \
					'A18T3WK7J16C1B:3IJXV6UZ1YR1KY4G6BFEG1DXR47RIC', \
					'A1ZTSCPETU3UJW:3XC1O3LBOTUGQEPEV3HM8W67X4RTLL', \
					'A1IUH9D7N69VS:3FTF2T8WLSQDHTSZ1BJ7Q7MB14SW9H', \
					'A2RCYLKY072XXO:3TAYZSBPLMG9ASQRWXURJVBCPLN2SH', \
					'AO1QGAUZ85T6L:3HWRJOOET6A15827PHPSLWK1MOYSEU', \
					'A2KBZ75HMJ7KVX:3EO896NRAX3AVO1ESI59SHTFTF3JTQ', \
					'A1F42D4L22K5ND:3ZSANO2JCGFTBM23KS9Y3E51YP4FSQ', \
					'A3GOOI75XOF24V:3TMFV4NEP9MD3O9PWJDTQBR0HZYW8I', \
					'A1HFNOKU591WCY:3XLBSAQ9Z5KDX59BX15UBFPNNHYZ7Q', \
					'A30AGR5KF8IEL:3EWIJTFFVPF14ZIVGF68BQEIRXP0ER', \
					'A2QIZ31TMHU0GD:3FUI0JHJPY6UBT1VAI7VUX8S3KH33W', \
					'AMW2XLD9443OH:37QW5D2ZRHUKW7SGCE3STMOFBMK8SX', \
					'A53S7J4JGWG38:3M0BCWMB8W4W5M7WZVX3HDH1P53WB1', \
					'A28RX7L0QZ993M:3SITXWYCNWHBUMCM90TPJWV8Y7AXBW', \
					'ALH1K6ZAQQMN7:3IXEICO793RY7TM78ZBKJDOA7ADT6Q']


def get_trial_list(datafile):
	# helper function to filter successful trials in human move data
	instance_data = []
	trial_data = []
	is_win = False
	# load datafile: human move data
	with open(datafile) as f:
		for line in f:
			instance_data.append(json.loads(line))
	# process each line of datafile, return only successful trials
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
		if move_num == '0': # indicates next trial has started
			if not is_win and len(trial_data)>0:
				trial_data.pop()
			trial_data.append([])
			trial_data[-1].append((car_to_move, move_to_position))
			is_win = False # reset flag in new trial
			continue
		trial_data[-1].append((car_to_move, move_to_position))
		if i == len(instance_data) - 1: # the last line of data file
			if not is_win:
				trial_data.pop()
	return trial_data


def get_solution(insdatafile, solutionfile): 
	# create solution (unique for now) for puzzle
	my_car_list, my_red = MAG.json_to_car_list(insdatafile)
	my_board, my_red = MAG.construct_board(my_car_list)
	board_str = MAG.board_to_str(my_board)
	print(board_str)
	sol_list = solution.main(board_str)
	print(sol_list)
	np.save(solutionfile, np.array(sol_list))


def prop_unsafe(insdatafile, datafile):
	# average proportion of unsafe moves in human move data
	trial_data = get_trial_list(datafile)
	total_unsafe_prop = 0
	# process successful trials
	for i in range(len(trial_data)): # each trial
		unsafe = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial mag
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		n_node, n_edge = MAG.get_mag_attr(new_car_list)
		for j in range(len(trial_data[i])): # each move
			my_board, my_red = MAG.board_move(new_car_list, \
										trial_data[i][j][0], trial_data[i][j][1])
			new_car_list = MAG.construct_mag(my_board, my_red)
			cur_node, cur_edge = MAG.get_mag_attr(new_car_list)
			if cur_edge > n_edge:
				unsafe += 1
			n_node = cur_node
			n_edge = cur_edge
		total_unsafe_prop += float(unsafe) / float(len(trial_data[i]) - 1) # ignore 1 redundant red move at the end
	return total_unsafe_prop / float(len(trial_data))


def prop_back_move(insdatafile, datafile):
	# average proportion of red-backward moving in human move data
	trial_data = get_trial_list(datafile)
	total_back_move = 0
	# process successful trials
	for i in range(len(trial_data)): # each trial
		back = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial mag
		for j in range(len(trial_data[i])): # each move
			my_car_list, new_red = MAG.move(my_car_list, \
											trial_data[i][j][0], trial_data[i][j][1])
			if new_red.start[0] < my_red.start[0]:
				back += 1
			my_red = new_red
		total_back_move += float(back) / float(len(trial_data[i]) - 1)
	return total_back_move / float(len(trial_data))


def avg_node_edge(insdatafile, datafile): 
	# average number of nodes and edges in human move data
	trial_data = get_trial_list(datafile)
	total_node = 0
	total_edge = 0
	# process successful trials
	for i in range(len(trial_data)): # each trial
		trial_node = 0
		trial_edge = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial mag
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		n_node, n_edge = MAG.get_mag_attr(new_car_list)
		trial_node += n_node
		trial_edge += n_edge
		for j in range(len(trial_data[i])): # each move
			my_car_list, my_red = MAG.move(my_car_list, \
										trial_data[i][j][0], trial_data[i][j][1])
			my_board, my_red = MAG.construct_board(my_car_list)
			new_car_list = MAG.construct_mag(my_board, my_red)
			n_node, n_edge = MAG.get_mag_attr(new_car_list)
			trial_node += n_node
			trial_edge += n_edge
		total_node += float(trial_node) / float(len(trial_data[i]) - 1) # ignore 1 redundant red move at the end
		total_edge += float(trial_edge) / float(len(trial_data[i]) - 1)
	return float(total_node) / float(len(trial_data)), \
			float(total_edge) / float(len(trial_data))



def avg_cycle(insdatafile, datafile):
	# return average number of cycles and average max cycle size in human moves data
	trial_data = get_trial_list(datafile)
	total_ncycle = 0
	total_maxcycle = 0
	# process successful trials
	for i in range(len(trial_data)): # each trial
		trial_ncycle = 0
		trial_maxcycle = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial MAG
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		ncycle, _, maxcycle = MAG.find_cycles(new_car_list) 
		trial_ncycle += ncycle
		trial_maxcycle += maxcycle
		for j in range(len(trial_data[i])): # each move
			my_car_list, my_red = MAG.move(my_car_list, \
										trial_data[i][j][0], trial_data[i][j][1])
			my_board, my_red = MAG.construct_board(my_car_list)
			new_car_list = MAG.construct_mag(my_board, my_red)
			ncycle, _, maxcycle = MAG.find_cycles(new_car_list) 
			trial_ncycle += ncycle
			trial_maxcycle += maxcycle
		total_ncycle += float(trial_ncycle) / float(len(trial_data[i]) - 1) # ignore 1 redundant red move at the end
		total_maxcycle += float(trial_maxcycle) / float(len(trial_data[i]) - 1)
	return float(total_ncycle) / float(len(trial_data)),\
			float(total_maxcycle) / float(len(trial_data))


def avg_node_cycle(insdatafile, datafile):
	# return average number of nodes in cycles for human moves data
	trial_data = get_trial_list(datafile)
	total_cnode = 0
	# process successful trials
	for i in range(len(trial_data)): # each trial
		trial_cnode = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial MAG
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		cnode, _ = MAG.num_nodes_in_cycle(new_car_list)  
		trial_cnode += cnode
		for j in range(len(trial_data[i])): # each move
			my_car_list, my_red = MAG.move(my_car_list, \
										trial_data[i][j][0], trial_data[i][j][1])
			my_board, my_red = MAG.construct_board(my_car_list)
			new_car_list = MAG.construct_mag(my_board, my_red)
			cnode, _ = MAG.num_nodes_in_cycle(new_car_list)
			trial_cnode += cnode
		total_cnode += float(trial_cnode) / float(len(trial_data[i]) - 1) # ignore 1 redundant red move at the end
	return float(total_cnode) / float(len(trial_data))


def avg_red_depth(insdatafile, datafile):
	# return average number of nodes in cycles for human move data
	trial_data = get_trial_list(datafile)
	total_depth = 0
	# process successful trials
	for i in range(len(trial_data)): # each trial
		trial_depth = 0
		my_car_list, my_red = MAG.json_to_car_list(insdatafile) # initial MAG
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		depth, _ = MAG.longest_path(new_car_list)
		trial_depth += depth
		for j in range(len(trial_data[i])): # each move
			my_car_list, my_red = MAG.move(my_car_list, \
										trial_data[i][j][0], trial_data[i][j][1])
			my_board, my_red = MAG.construct_board(my_car_list)
			new_car_list = MAG.construct_mag(my_board, my_red)
			depth, _ = MAG.longest_path(new_car_list)
			trial_depth += depth
		total_depth += float(trial_depth) / float(len(trial_data[i]) - 1) # ignore 1 redundant red move at the end
	return float(total_depth) / float(len(trial_data))





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

y_human = []
y_opt = []
data = []
all_human_len = [] # all human length for every puzzle
optimal_list = []

# preprocess human & optimal len dict
for i in range(len(valid_subjects)): # initialize
	all_human_len.append([])
	optimal_list.append([])
	for j in range(len(all_instances)):
		all_human_len[i].append(0)
		optimal_list[i].append(0)
with open(len_file) as f:
	for line in f:
		data.append(json.loads(line))
for i in range(0, len(data)): # iterate through every subject trial
	line = data[i]
	instance = line['instance']
	subject = line['subject']
	assign = line['assignment']
	complete = line['complete']
	human_len = int(line['human_length'])
	opt_len = int(line['optimal_length'])
	if complete == 'False':
		continue
	elif (subject + ':' + assign) in valid_subjects:
		ins_index1 = valid_subjects.index(subject + ':' + assign)
		ins_index2 = all_instances.index(instance)
		all_human_len[ins_index1][ins_index2] = human_len
		optimal_list[ins_index1][ins_index2] = opt_len
for i in range(len(valid_subjects)):
	for j in range(len(all_instances)): # flatten human len
		if all_human_len[i][j] == 0:
			continue
		else:
			y_human.append(all_human_len[i][j])
			y_opt.append(optimal_list[i][j])



dict_list = [[] for _ in range(num_features)]
y_list = [[] for _ in range(num_features)]
# process mag attributes
count = 0
for j in range(len(valid_subjects)):
	print(valid_subjects[j])
	for i in range(len(all_instances)):
		# construct mag
		cur_ins = all_instances[i]
		ins_file = ins_dir + all_instances[i] + '.json'
		# move_file = move_dir + all_instances[i] + '_moves.json'
		sol_file = move_dir + all_instances[i] + '_solution.npy'	
		# MAG features from solution file
		if all_human_len[j][i] == 0:
			continue
		dict_list[0].append(prop_unsafe_solution(ins_file, sol_file))
		# print('prop unsafe solution: ', dict_list[0][cur_ins])
		dict_list[1].append(prop_back_move_solution(ins_file, sol_file))
		# print('prop backmove solution: ', dict_list[1][cur_ins])
		one, two = avg_node_edge_solution(ins_file, sol_file)
		dict_list[2].append(one)
		dict_list[3].append(two)
		# print('avg node, edge solution: ' + str(dict_list[2][cur_ins]) \
				# + ', ' + str(dict_list[3][cur_ins]))
		one, two = avg_cycle_solution(ins_file, sol_file)
		dict_list[4].append(one)
		dict_list[5].append(two)
		# print('avg ncycle, maxcycle solution: ' + str(dict_list[4][cur_ins]) \
				# + ', ' + str(dict_list[5][cur_ins]))
		dict_list[6].append(avg_node_cycle_solution(ins_file, sol_file))
		dict_list[7].append(avg_red_depth_solution(ins_file, sol_file))
		one, two = node_edge_rate(ins_file, y_opt[count])
		dict_list[8].append(one)
		dict_list[9].append(two)
		# print('node rate, edge rate:' + str(dict_list[8][cur_ins]) + ' ' + str(dict_list[9][cur_ins]))
		count += 1

print(len(y_human))
print(len(y_opt))
print(len(dict_list[3]))

# generate value lists
for i in range(len(y_human)):
	for j in range(0, num_features):
		y_list[j].append(dict_list[j][i])
# save data
np.save(data_out + 'y_human.npy', y_human) # mean human len
np.save(data_out + 'y_opt.npy', y_opt) # opt len
for i in range(2,len(feature_list)):
	np.save(data_out + feature_list[i] + '.npy', y_list[i-2])


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
		axarr[i,j].scatter(y_list[count], y_human)
		axarr[i,j].set_xlabel(label_list[count+2], fontsize=20, fontweight='bold')
		t = 'rho=%s, p=%s'%(format(corr_list[count], '.3f'), format(p_list[count], '.2g'))
		axarr[i,j].set_title(t,y=0.85,fontsize=17)
		axarr[i,j].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
		axarr[i,j].xaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
		axarr[i,j].tick_params(axis='both', labelsize=17)
		axarr[i,j].grid(True, axis='y', alpha=0.3)
		count += 1

plt.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5, rect=None) 
plt.suptitle('human_len vs dynamic MAG', y=0.999, fontsize=22, fontweight='bold')
# plt.show()
plt.savefig(scatter_out)
plt.close()



