# plot value function along optinal solution paths
# number of cars in each level
# automatically generate plot for each level, max level number = 7
# py27
'''
value = w0R * num_cars(red, right)
		+ w1 * num_cars{MAG-level 1} 
		+ w2 * num_cars{MAG-level 2}  
		+ w3 * num_cars{MAG-level 3} 
		+ w4 * num_cars{MAG-level 4} 
		+ w5 * num_cars{MAG-level 5} 
		+ w6 * num_cars{MAG-level 6}
		+ w7 * num_cars{MAG-level 7}  
		+ noise
'''
import MAG
import numpy as np
import sys, random
import matplotlib.pyplot as plt

# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
move_dir = '/Users/chloe/Documents/RushHour/exp_data/'
fig_out = '/Users/chloe/Desktop/model-num-cars-level-'
# value function initial parameters and weights
num_parameters = 8
noise = 0
value = 0
level_selected = 7
offset_magnitide = 0.22
ylim = 4

# each parameter
for w_idx in range(num_parameters):
	
	[w0R, w1, w2, w3, w4, w5, w6, w7] = [0,0,0,0,0,0,0,0]
	weights = [w0R, w1, w2, w3, w4, w5, w6, w7]
	weights[w_idx] = 1
	print(weights)

	# each puzzle
	for i in range(len(all_instances)): 
		
		random_offset = np.random.uniform(low=-0.10, high=0.10)
		
		# read solution file
		cur_ins = all_instances[i]
		ins_file = ins_dir + cur_ins + '.json'
		sol_file = move_dir + cur_ins + '_solution.npy'	
		sol_list = np.load(sol_file)
		
		# only process selected level only
		if len(sol_list)+1 != level_selected: 
			continue
		
		# construct initial MAG
		my_car_list, my_red = MAG.json_to_car_list(ins_file)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)

		debug = 0
		value_seq = []

		# each move in solution, omit last step
		for move in sol_list: 
		
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

		# plot value sequence for this puzzle
		plt.plot(np.arange(len(value_seq)), np.array(value_seq, dtype=np.float32)+random_offset, \
				'-o', markersize=2,
				color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))


	# save plot for this parameter, across puzzles
	plt.ylim(top=ylim+offset_magnitide)
	plt.yticks(np.arange(0, ylim+1, dtype=np.int32))
	plt.xlabel('Move number along optimal solution')
	plt.title('Num_cars, level ' + str(level_selected) + ', ' + str(weights))	
	plt.savefig(fig_out+str(level_selected)+'-'+str(w_idx)+'.png')
	plt.close()


