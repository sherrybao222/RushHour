# plot value function along optinal solution paths
# average number of cars in each level across all puzzles
# automatically generate plot for each level, max level number = 7
# py27
'''
value = avg_across_puzzles(
		w0R * num_cars(red, right)
		+ w1 * num_cars{MAG-level 1} 
		+ w2 * num_cars{MAG-level 2}  
		+ w3 * num_cars{MAG-level 3} 
		+ w4 * num_cars{MAG-level 4} 
		+ w5 * num_cars{MAG-level 5} 
		+ w6 * num_cars{MAG-level 6}
		+ w7 * num_cars{MAG-level 7} )
		+ noise
'''
import MAG
import numpy as np
import sys, random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
move_dir = '/Users/chloe/Documents/RushHour/exp_data/'
fig_out = '/Users/chloe/Desktop/model-avg-num-cars-level-'
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',\
			# 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
color_gradients = [0.125, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
# value function initial parameters and weights
num_parameters = 8
noise = 0
value = 0
# length_selected = 7
#offset_magnitide = 0.22
ylim = 0
xlim = 1
cmap = matplotlib.cm.get_cmap('Blues')

for length_selected in [7, 11, 14, 16]: # each puzzle length
	# each parameter
	for w_idx in range(num_parameters):
		
		[w0R, w1, w2, w3, w4, w5, w6, w7] = [0,0,0,0,0,0,0,0]
		weights = [w0R, w1, w2, w3, w4, w5, w6, w7]
		weights[w_idx] = 1
		print(weights)

		num_puzzles = 0
		values_puz = np.zeros(length_selected-1)
		cache = []

		# each puzzle
		for i in range(len(all_instances)): 
			
			#random_offset = np.random.uniform(low=-0.10, high=0.10)
			
			# read solution file
			cur_ins = all_instances[i]
			ins_file = ins_dir + cur_ins + '.json'
			sol_file = move_dir + cur_ins + '_solution.npy'	
			sol_list = np.load(sol_file)
			
			# only process selected level only
			if len(sol_list)+1 != length_selected: 
				continue
			num_puzzles += 1
			
			# construct initial MAG
			my_car_list, my_red = MAG.json_to_car_list(ins_file)
			my_board, my_red = MAG.construct_board(my_car_list)
			new_car_list = MAG.construct_mag(my_board, my_red)
			n_node, n_edge = MAG.get_mag_attr(new_car_list)

			debug = 0
			value_seq = []

			# print('length of solution list: '+ str(len(sol_list)))
			# each move in solution, omit last step
			for move in sol_list: 

				# print(MAG.board_to_str(my_board))
			
				debug += 1
				value = 0

				# number of cars at the top level (red only)
				value += w0R * 1 

				# each level
				for j in range(num_parameters - 1): 
					level = j+1
					new_car_list = MAG.clean_level(new_car_list)
					new_car_list = MAG.assign_level(new_car_list, my_red)
					cars_from_level = MAG.get_cars_from_level2(new_car_list, level)
					# print('cars from level ', level)
					value += weights[level] * (len(cars_from_level))

				# print(value)
				value_seq.append(value)

				# make a move
				car_to_move = move[0]
				if car_to_move == 'R':
					car_to_move = 'r'
				move_by = int(move[2:])
				my_car_list, my_red = MAG.move_by(my_car_list, \
												car_to_move, move_by)
				my_board, my_red = MAG.construct_board(my_car_list)
				new_car_list = MAG.construct_mag(my_board, my_red)

			values_puz += np.array(value_seq, dtype=np.float32)
			cache.append(np.array(value_seq, dtype=np.float32))
			# sys.exit()

		values_puz = values_puz / num_puzzles
		std = np.std(cache, axis=0)
		# print(std)
		if max(values_puz) > ylim:
			ylim = max(values_puz)
		# plot this parameter
		rgba = cmap(color_gradients[w_idx])
		plt.plot(np.arange(len(values_puz)), np.array(values_puz, dtype=np.float32), \
				'-o', markersize=5, linewidth=5,\
				color=rgba, label=str(w_idx))
		# plt.errorbar(np.arange(len(values_puz)), np.array(values_puz, dtype=np.float32),\
				# std, linestyle='None', color=colors[w_idx], alpha=0.5)
		#for x, y in zip(np.arange(len(values_puz)), np.array(values_puz, dtype=np.float32)):
		#	plt.text(x, y, str(w_idx), color="black", fontsize=6)

	# all parameters in the same plot
	fig = plt.gcf()
	fig.set_size_inches((length_selected-1)*1.8, 15)
	# plt.ylim(top=ylim+0.1)
	plt.ylim(top=3.0)
	plt.xticks(np.arange(length_selected-1), np.arange(1, length_selected))
	# plt.legend(prop={'size': 40})
	plt.xticks(fontsize=60)
	plt.yticks(fontsize=60)
	plt.xlabel('Move Number', fontsize=50)
	plt.grid(linestyle='--', alpha=0.3)
	# plt.title('Avg Num_cars at each level, length-' \
	# 		+ str(length_selected-1),\
	# 		fontsize=30)
	plt.title('Length-' \
			+ str(length_selected-1)+' Puzzles',\
			fontsize=50)	
	plt.savefig(fig_out+str(length_selected-1)+'.png')
	plt.close()


