# for each move in trialdata, generate attributes
# include difference between current length and optlen, 
# each row records the decision made already
# need to run with python27
import MAG
import solution
import sys, csv
import pandas as pd
import numpy as np
import time

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
out_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed_bin5.csv'
instance_folder = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'


#################################### MAIN PROGRAM ###############################

move_data = pd.read_csv(moves_file)
diff_out = []

prev_event = ''
instance_optlen = ''

for i in range(len(move_data)):
	# first line
	if i == 0: 
		diff_out.append(np.nan)
		row = move_data.loc[i, :]
		prev_event = row['event']
		prev_optlen = row['optlen']
		instance_optlen = prev_optlen
		continue
	row = move_data.loc[i, :]
	cur_event = row['event']
	cur_optlen = row['optlen']

	if cur_event == 'start':
		diff_out.append(np.nan)
		instance_optlen = cur_optlen
	else:
		diff_out.append(cur_optlen - instance_optlen)

	prev_event = cur_event
	prev_optlen = cur_optlen

print('diflen out ', len(diff_out))


# save result
with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('d'))
	for j in range(len(diff_out)):
		writer.writerow([diff_out[j]])

















