# generate binary attributes for each move
# include restart decision, surrender decision, 
# error detection, current further than initial
# each row records the decision made already
# need to run with python365
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
out_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed_bin.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'


#################################### MAIN PROGRAM ###############################

move_data = pd.read_csv(moves_file)
restart_out = []
surrender_out = []
error_out = []
further_out = []


prev_subject = ''
prev_instance = ''
prev_move_number = ''
prev_trial_number = ''
prev_event = ''
prev_optlen = ''
instance_optlen = ''

counter = 0
start_time = time.time()

for i in range(len(move_data)):
	# first line
	if i == 0: 
		error_out.append(0)
		further_out.append(0)
		row = move_data.loc[i, :]
		prev_subject = row['subject']
		prev_instance = row['instance']
		prev_move_number = row['move_num']
		prev_event = row['event']
		prev_optlen = row['optlen']
		instance_optlen = prev_optlen
		# print('\nfinish row ', i)
		continue
	row = move_data.loc[i, :]
	cur_subject = row['subject']
	cur_instance = row['instance']
	cur_move_number = row['move_num']
	cur_event = row['event']
	cur_optlen = row['optlen']
	# if current is further than initial
	if cur_event == 'start':
		further_out.append(0)
		instance_optlen = cur_optlen
	elif cur_subject == prev_subject and cur_instance == prev_instance and cur_optlen > instance_optlen:
		further_out.append(1)
	else:
		further_out.append(0)
	# error detection
	if cur_event == 'start':
		error_out.append(0)
	elif cur_subject == prev_subject and cur_instance == prev_instance and cur_optlen == prev_optlen - 1:
		error_out.append(0)
	else:
		error_out.append(1)
	# win
	if cur_event == 'start' and cur_instance != prev_instance and prev_optlen == 1:
		print('win found row ', i)
		restart_out.append(0)
		surrender_out.append(0)
		prev_subject = cur_subject
		prev_instance = cur_instance
		prev_move_number = cur_move_number
		prev_event = cur_event
		prev_optlen = cur_optlen
		# print('\nfinish row ', i)
		continue
	# restart
	if cur_event == 'start' and cur_subject == prev_subject and cur_instance == prev_instance:
		if cur_move_number == 0:
			restart_out.append(1)
			surrender_out.append(0)
			prev_subject = cur_subject
			prev_instance = cur_instance
			prev_move_number = cur_move_number
			prev_event = cur_event
			prev_optlen = cur_optlen
			# print('\nfinish row ', i)
			continue
	# surrender
	if cur_event == 'start' and cur_subject == prev_subject and cur_instance != prev_instance:
		if cur_move_number == 0:
			surrender_out.append(1)
			restart_out.append(0)
			prev_subject = cur_subject
			prev_instance = cur_instance
			prev_move_number = cur_move_number
			prev_event = cur_event
			prev_optlen = cur_optlen
			# print('\nfinish row ', i)
			continue
	# else
	restart_out.append(0)
	surrender_out.append(0)
	prev_subject = cur_subject
	prev_instance = cur_instance
	prev_move_number = cur_move_number
	prev_event = cur_event
	prev_optlen = cur_optlen
	# print('\nfinish row ', i)	

print('restart out ', len(restart_out))
print('surrender out ', len(surrender_out))
print('error out ', len(error_out))
print('further out ', len(further_out))

# save result
with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('restart', 'surrender', 'error', 'further'))
	for j in range(len(restart_out)):
		writer.writerow([restart_out[j], surrender_out[j], error_out[j], further_out[j]])

















