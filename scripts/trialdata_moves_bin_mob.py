# generate binary attributes for each move
# include mobility reduced
# each row records the decision made already
# need to run with python365
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
out_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed_bin3.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'


#################################### MAIN PROGRAM ###############################

move_data = pd.read_csv(moves_file)
reduced_mob_out = [] # only mobility descreased
tie_mob = [] # tie mobility (equal mobility as last move)


prev_subject = ''
prev_instance = ''
prev_trial_number = ''
prev_event = ''
prev_mobility = ''


for i in range(len(move_data)):
	# first line
	if i == 0: 
		reduced_mob_out.append(0)
		tie_mob.append(0)
		row = move_data.loc[i, :]
		prev_subject = row['subject']
		prev_instance = row['instance']
		prev_event = row['event']
		prev_mobility = row['mobility']
		continue
	row = move_data.loc[i, :]
	cur_subject = row['subject']
	cur_instance = row['instance']
	cur_event = row['event']
	cur_mobility = row['mobility']
	# error detection
	if cur_event == 'start':
		reduced_mob_out.append(0)
		tie_mob.append(0)
	elif cur_subject == prev_subject and cur_instance == prev_instance and cur_mobility > prev_mobility:
		reduced_mob_out.append(0) # increased mob
		tie_mob.append(0)
	elif cur_subject == prev_subject and cur_instance == prev_instance and cur_mobility == prev_mobility:
		tie_mob.append(1) # tied 
		reduced_mob_out.append(0)
	else: # reduced mob
		tie_mob.append(0)
		reduced_mob_out.append(1)
	prev_subject = cur_subject
	prev_instance = cur_instance
	prev_event = cur_event
	prev_mobility = cur_mobility

print('mobility out ', len(reduced_mob_out))
print('tie mob ', len(tie_mob))

# save result
with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('m', 't'))
	for j in range(len(reduced_mob_out)):
		writer.writerow([reduced_mob_out[j], tie_mob[j]])

















