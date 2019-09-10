# plot value function along optinal solution paths
# average number of edges in each level across all puzzles
# automatically generate plot for each level, max level number = 7
# py27
'''
value = avg_across_puzzles(
		w0R * num_cars(red)
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
import sys, random, os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator
import pandas as pd

# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
move_dir = '/Users/chloe/Documents/RushHour/exp_data/'
fig_out = '/Users/chloe/Desktop/Human-'
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',\
			# 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
# color_gradients = [0.125, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
color_gradients = [0.875, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
# value function initial parameters and weights
num_parameters = 8
noise = 0
value = 0
# length_selected = 7
#offset_magnitide = 0.22
ylim = 0
xlim = 1
cmap = matplotlib.cm.get_cmap('Blues')

# 54, 65; 21, 53; 445, 470; 718, 732
trial_start = 1157
trial_end = 1188
sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')

# construct initial node
first_line = sub_data.loc[trial_start-2,:]
instance = first_line['instance']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
ins_file = ins_dir + instance + '.json'
# construct initial MAG
my_car_list, my_red = MAG.json_to_car_list(ins_file)
my_board, my_red = MAG.construct_board(my_car_list)
print('len initial car list ', len(my_car_list))
print(MAG.board_to_str(my_board))
print(my_red.tag)
new_car_list = MAG.construct_mag(my_board, my_red)
print('new initial car list ', len(new_car_list))


debug = 0

# each parameter
for w_idx in range(1):
	
	# [w0R, w1, w2, w3, w4, w5, w6, w7] = [0,0,0,0,0,0,0,0]
	[w0R, w1, w2, w3, w4, w5, w6, w7] = [1,6,6,4,3,2,1,1]
	weights = [w0R, w1, w2, w3, w4, w5, w6, w7]
	# weights[w_idx] = 1
	value_seq = []
	# construct initial MAG
	my_car_list, my_red = MAG.json_to_car_list(ins_file)
	my_board, my_red = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)

	# every move in the trial
	for i in range(trial_start-1, trial_end-1):

		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = int(row['move'])

		# calculate value at this position
		value = 0
		# number of cars at the top level (red only)
		value += w0R * 1 

		# each level
		for j in range(num_parameters - 2): 
			level = j+1
			new_car_list = MAG.clean_level(new_car_list)
			new_car_list = MAG.assign_level(new_car_list, my_red)
			cars_from_level = MAG.get_cars_from_level2(new_car_list, level)
			# print(len(cars_from_level))
			value += weights[level] * (len(cars_from_level))

		value_seq.append(value)
		print(value)
		print(MAG.board_to_str(my_board))

		print('piece move_to', piece, move_to)
		# make actual move by human 
		my_car_list, my_red = MAG.move(my_car_list, piece, move_to)
		my_board, my_red = MAG.construct_board(my_car_list)
		new_car_list = MAG.construct_mag(my_board, my_red)

		debug += 1
		# if debug == 20:
		# 	sys.exit()


	# plot this parameter
	rgba = cmap(color_gradients[w_idx])
	plt.plot(np.arange(len(value_seq)), np.array(value_seq, dtype=np.float32), \
			'-o', markersize=5, linewidth=5,\
			color=rgba, label=str(w_idx))

# all parameters in the same plot
fig = plt.gcf()
fig.set_size_inches((len(value_seq)-1)*1.8, 15)
# plt.ylim(top=ylim+0.1)
# plt.ylim(top=3.0)
plt.xticks(np.arange(len(value_seq)-1), np.arange(1, len(value_seq)))
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.xlabel('Move Number', fontsize=50)
plt.grid(linestyle='--', alpha=0.3)
plt.title('HumanLength-' \
		+ str(len(value_seq)-1)+' Puzzles',\
		fontsize=50)	
plt.savefig(fig_out+str(len(value_seq)-1)+'.png')
plt.close()

