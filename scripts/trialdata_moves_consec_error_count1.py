# generate attributes for each move
# consecutive error
# consecutive error made closer than initial
# consecutive error made further than initial
# each row records the decision made already
# need to run with python27
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
out_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed_bin2.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'


#################################### MAIN PROGRAM ###############################

move_data = pd.read_csv(moves_file)
consec_error_out = []
consec_error_closer = []
consec_error_further = []


prev_subject = ''
prev_instance = ''
prev_move_number = ''
prev_trial_number = ''
prev_event = ''
prev_optlen = ''
instance_optlen = ''
ins_consec_error = 0
ins_consec_error_closer = 0
ins_consec_error_further = 0

counter = 0
start_time = time.time()

for i in range(len(move_data)):
	# first line
	if i == 0: 
		consec_error_out.append(0)
		consec_error_closer.append(0)
		consec_error_further.append(0)
		row = move_data.loc[i, :]
		prev_subject = row['subject']
		prev_instance = row['instance']
		prev_event = row['event']
		prev_error = row['error']
		prev_further = row['further']
		instance_optlen = prev_optlen
		# print('\nfinish row ', i)
		continue
	row = move_data.loc[i, :]
	cur_subject = row['subject']
	cur_instance = row['instance']
	cur_event = row['event']
	cur_error = row['error']
	cur_further = row['further']
	# start
	if cur_event == 'start':
		ins_consec_error = 0
		ins_consec_error_closer = 0
		ins_consec_error_further = 0
	# error found
	elif cur_subject == prev_subject and cur_instance == prev_instance and cur_error == 1:
		ins_consec_error += 1
		if cur_further == 1:
			ins_consec_error_further += 1
			ins_consec_error_closer = 0
		elif cur_further == 0:
			ins_consec_error_closer += 1
			ins_consec_error_further = 0
	# no error found
	else:
		ins_consec_error = 0
		ins_consec_error_closer = 0
		ins_consec_error_further = 0

	consec_error_out.append(ins_consec_error)
	consec_error_closer.append(ins_consec_error_closer)
	consec_error_further.append(ins_consec_error_further)

	prev_subject = cur_subject
	prev_instance = cur_instance
	prev_event = cur_event
	prev_error = cur_error
	prev_further = cur_further
	# print('\nfinish row ', i)	

print('consec error out ', len(consec_error_out))
print('consec error further ', len(consec_error_further))
print('consec error closer ', len(consec_error_closer))


# save result
with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('consec_error', 'consec_error_closer', 'consec_error_further'))
	for j in range(len(consec_error_out)):
		writer.writerow([consec_error_out[j], consec_error_closer[j], consec_error_further[j]])

















