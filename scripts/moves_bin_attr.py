# generate binary attributes for each move
# include restart decision, surrender decision
# each row records the decision to be made
# need to run with python365
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time

moves_file = '/Users/chloe/Documents/RushHour/exp_data/moves_valid.csv'
out_file = '/Users/chloe/Documents/RushHour/exp_data/moves_bin_attr.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
header =  ['worker', 'assignment', 'instance', \
			'optlen', 'move_number', 'move', \
			'meta_move', 'rt', 'trial_number']

move_data = pd.read_csv(moves_file)
restart_out = []
surrender_out = []

prev_subject = ''
prev_instance = ''
prev_move_number = ''
prev_trial_number = ''
prev_meta_move = ''

counter = 0
start_time = time.time()

for i in range(len(move_data)):
	# first line
	if i == 0: 
		row = move_data.loc[i, :]
		prev_subject = row['worker'] + ':' + row['assignment']
		prev_instance = row['instance']
		prev_move_number = row['move_number']
		prev_trial_number = row['trial_number']
		prev_meta_move = row['meta_move']
		print('\nfinish row ', i)
		continue
	row = move_data.loc[i, :]
	cur_subject = row['worker'] + ':' + row['assignment']
	cur_instance = row['instance']
	cur_move_number = row['move_number']
	cur_trial_number = row['trial_number']
	cur_meta_move = row['meta_move']
	# win
	if cur_trial_number == 0 or cur_meta_move == 'win':
		restart_out.append(0)
		surrender_out.append(0)
		prev_subject = cur_subject
		prev_instance = cur_instance
		prev_move_number = cur_move_number
		prev_trial_number = cur_trial_number
		prev_meta_move = cur_meta_move
		print('\nfinish row ', i)
		continue
	# restart
	if cur_subject == prev_subject and cur_instance == prev_instance:
		if cur_move_number == 0 and prev_trial_number != 0 and cur_trial_number == prev_trial_number + 1:
			restart_out.append(1)
			surrender_out.append(0)
			prev_subject = cur_subject
			prev_instance = cur_instance
			prev_move_number = cur_move_number
			prev_trial_number = cur_trial_number
			prev_meta_move = cur_meta_move
			print('\nfinish row ', i)
			continue
	# surrender
	if cur_subject == prev_subject and cur_instance != prev_instance:
		if cur_move_number == 0 and prev_trial_number != 0:
			surrender_out.append(1)
			restart_out.append(0)
			prev_subject = cur_subject
			prev_instance = cur_instance
			prev_move_number = cur_move_number
			prev_trial_number = cur_trial_number
			prev_meta_move = cur_meta_move
			print('\nfinish row ', i)
			continue
	# else
	restart_out.append(0)
	surrender_out.append(0)
	prev_subject = cur_subject
	prev_instance = cur_instance
	prev_move_number = cur_move_number
	prev_trial_number = cur_trial_number
	prev_meta_move = cur_meta_move
	print('\nfinish row ', i)	

print('restart out ', len(restart_out))
print('surrender out ', len(surrender_out))

# save result
with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('restart', 'surrender'))
	for j in range(len(restart_out)):
		writer.writerow([restart_out[j], surrender_out[j]])

















