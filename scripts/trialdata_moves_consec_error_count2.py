# generate attributes for each move
# consecutive error made cross (closer and further)
# each row records the decision made already
# need to run with python27
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
consec_error_cross = [0] * len(move_data)

# closer followed by further
flag = False
for i in range(len(move_data)):
	row = move_data.loc[i, :]
	consec_error = row['consec_error']
	consec_error_closer = row['consec_error_closer']
	consec_error_further = row['consec_error_further']
	if consec_error_closer != 0:
		flag = True
	else: 
		if consec_error_further != 0 and flag:
			consec_error_cross[i] = consec_error
			flag = True
		else:
			flag = False

flag = False
i = len(move_data) - 1
while i >= 0:
	row = move_data.loc[i, :]
	consec_error = row['consec_error']
	consec_error_closer = row['consec_error_closer']
	consec_error_further = row['consec_error_further']
	if consec_error_further != 0:
		flag = True
	else: 
		if consec_error_closer != 0 and flag:
			consec_error_cross[i] = consec_error
			flag = True
		else:
			flag = False
	i -= 1


print('consec_error_cross ', consec_error_cross)


# save result
with open(out_file, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(('cross'))
	for j in range(len(consec_error_cross)):
		writer.writerow([consec_error_cross[j]])












