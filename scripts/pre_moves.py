# filter trialdata by valid subject list
# run with py27

import csv
import numpy as np
import pandas as pd

valid_subject = np.load('/Users/chloe/Documents/RushHour/exp_data/valid_sub.npy')
in_csv_file = '/Users/chloe/Documents/RushHour/exp_data/trialdata.csv'
out_csv_file = '/Users/chloe/Documents/RushHour/exp_data/trialdata_valid.csv'

out_data = []

# filter data
with open(in_csv_file, 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		cur_subject = row[0]
		if cur_subject not in valid_subject:
			print(cur_subject)
			continue
		out_data.append(row)

# write out_data
with open(out_csv_file, mode='w') as outfile:
	csv_writer = csv.writer(outfile)
	for row in out_data:
		csv_writer.writerow(row)

print('len out_data ', len(out_data))

