import MAG
import numpy as np
import sys, random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator

# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
data_dir = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
fig_dir = '/Users/chloe/Desktop/diffoptlen_traces-'
puzzles = [2, 30, 60]

move_data = pd.read_csv(data_dir)
prev_sub = ''

# each puzzle
for puzzle in puzzles:
	# load puzzle name
	puzzle = all_instances[puzzle]
	all_trials = []
	trial = []
	# visualize board
	ins_file = ins_dir + puzzle + '.json'
	my_car_list, my_red = MAG.json_to_car_list(ins_file)
	my_board, my_red = MAG.construct_board(my_car_list)
	print(puzzle)
	print(MAG.board_to_str(my_board))

	# go through entire data file
	for i in range(len(move_data)):
		row = move_data.loc[i, :]
		instance = row['instance']
		# continue processing iff instance matches puzzle name
		if instance != puzzle:
			continue
		subject = row['subject']
		event = row['event']
		diffoptlen = row['diffoptlen']
		initial = row['initial']
		# start a new array for a new trial
		if event == 'start':
			all_trials.append(trial)
			trial = []
		# append data
		trial.append(diffoptlen)
	# save the last trial
	all_trials.append(trial)

	# plot traces for this puzzle
	x_max = 0
	ax = figure().gca()

	for trial in all_trials:
		ax.plot(np.arange(len(trial)), \
				np.array(trial+initial-1, dtype=np.float32), \
				'-', linewidth=5)
		if len(trial) > x_max:
			x_max = len(trial)
	fig = plt.gcf()
	fig.set_size_inches(x_max/4, 10.5)
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	plt.axvline(x=int(initial)-1, color='gray', linestyle='--')
	plt.grid(linestyle='--', alpha=0.3)
	plt.xticks(fontsize=40)
	plt.yticks(fontsize=40)
	plt.xlabel('Move Number', fontsize=40)
	plt.ylabel('Distance to Goal', fontsize=40)
	plt.title('Example Length-'+str(initial-1)+' Puzzle', fontsize=40)
	plt.savefig(fig_dir+puzzle+'.png')
	plt.close()




















