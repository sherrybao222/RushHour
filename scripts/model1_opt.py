# plot value function along optinal solution paths
# py27
# automatically generate plot for each level
'''
This value function is now abandoned

value = w0R * Mobility(red, right) + w0L * Mobility(red, left) 
		+ w1 * sum_{MAG-level 1 cars} Mobility(car)  
		+ w2 * sum_{MAG-level 2 cars} Mobility(car) 
		+ w3 * sum_{MAG-level 3 cars} Mobility(car) 
		+ w4 * sum_{MAG-level 4 cars} Mobility(car)
		+ ... + noise
'''
import MAG
import numpy as np
import sys, random
import matplotlib.pyplot as plt

# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
move_dir = '/Users/chloe/Documents/RushHour/exp_data/'
fig_out = '/Users/chloe/Desktop/model-sum-mob-level'
# value function initial parameters and weights
num_parameters = 6
noise = 0
value = 0
level_selected = 7
offset_magnitide = 0.22
ylim=10

for w_idx in range(num_parameters):
	[w0R, w0L, w1, w2, w3, w4] = [0,0,0,0,0,0]
	weights = [w0R, w0L, w1, w2, w3, w4]
	weights[w_idx] = 1
	for i in range(len(all_instances)): # each puzzle
		random_offset = np.random.uniform(low=-offset_magnitide, high=offset_magnitide)
		# read solution file
		cur_ins = all_instances[i]
		ins_file = ins_dir + cur_ins + '.json'
		sol_file = move_dir + cur_ins + '_solution.npy'	
		sol_list = np.load(sol_file)
		if len(sol_list)+1 != level_selected:
			continue
		# construct initial MAG
		my_car_list, my_red = MAG.json_to_car_list(ins_file)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)
		n_node, n_edge = MAG.get_mag_attr(new_car_list)

		debug = 0
		value_seq = []
		
		for move in sol_list: # each step in solution, omit last step
			debug += 1
			# calculate value function
			value = 0
			# print('---------------------------')
			# print(MAG.board_to_str(my_board))
			red_mob_R, red_mob_L = MAG.red_mobility(my_board, my_red)
			value += w0R * red_mob_R + w0L * red_mob_L
			MAG.visualize_mag(new_car_list, '/Users/chloe/Desktop/test')
			for j in range(num_parameters - 2): # each level
				level = j + 1
				new_car_list = MAG.clean_level(new_car_list)
				new_car_list = MAG.assign_level(new_car_list, my_red)
				cars_from_level = MAG.get_cars_from_level2(new_car_list, level)
				# print('cars from level ', level)
				for cur_car in cars_from_level: # each car in tat level
					mobility =  MAG.get_car_mobility(my_board, cur_car)
					# print(cur_car.tag, mobility)
					value += weights[j + 2] * mobility
			# print(value)
			value_seq.append(value)
			plt.plot(np.arange(len(value_seq)), np.array(value_seq, dtype=np.float32)+random_offset, \
				'-o', markersize=2,
				color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
			# make a move
			car_to_move = move[0]
			if car_to_move == 'R':
				car_to_move = 'r'
			move_by = int(move[2:])
			my_car_list, my_red = MAG.move_by(my_car_list, \
											car_to_move, move_by)
			my_board, my_red = MAG.construct_board(my_car_list)
			new_car_list = MAG.construct_mag(my_board, my_red)

	plt.ylim(top=ylim+offset_magnitide)
	plt.yticks(np.arange(0, ylim+1, dtype=np.int32))
	plt.title('Sum mobility, level ' + str(level_selected) + ', ' + str(weights))	
	plt.savefig(fig_out+str(level_selected)+'-'+str(w_idx+1)+'.png')
	plt.close()

