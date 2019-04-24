# generate attributes for each move
# consecutive mobility reduced
# each row records the decision made already
# need to run with python27
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
out_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed_bin4.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'


#################################### MAIN PROGRAM ###############################

move_data = pd.read_csv(moves_file)
consec_mobred_out = []

prev_subject = ''
prev_instance = ''
prev_event = ''
prev_mobred = ''
ins_consec_mobred = 0


for i in range(len(move_data)):
	if i == 0: 
		consec_mobred_out.append(0)
		row = move_data.loc[i, :]
		prev_subject = row['subject']
		prev_instance = row['instance']
		prev_event = row['event']
		prev_mobred = row['mobility_reduced']
		continue
	row = move_data.loc[i, :]
	cur_subject = row['subject']
	cur_instance = row['instance']
	cur_event = row['event']
	cur_mobred = row['mobility_reduced']
	# start
	if cur_event == 'start':
		ins_consec_error = 0
	# mobred found
	elif cur_subject == prev_subject and cur_instance == prev_instance and cur_mobred == 1:
		ins_consec_mobred += 1
	# no mobred found
	else:
		ins_consec_mobred = 0

	consec_mobred_out.append(ins_consec_mobred)

	prev_subject = cur_subject
	prev_instance = cur_instance
	prev_event = cur_event
	prev_mobred = cur_mobred

print('consec mobred out ', len(consec_mobred_out))


# save result
with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('c'))
	for j in range(len(consec_mobred_out)):
		writer.writerow([consec_mobred_out[j]])

















