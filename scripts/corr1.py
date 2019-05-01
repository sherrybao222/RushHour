# PLOTTING: characterize trialdata each move
# probability of restart as a function of optlen
# probability of restart as a function of consecutive errors
# probability of restart as a function of mobility
# probability of restart as a function of consec_mobility_reduced
# consecutive errors: overall consec_error, consec_error_closer, consec_error_further
# density histograms of optlen
# density histograms of consec errors
# density histograms of consec_mobility_reduced
# need to run with python27
import sys, csv, math
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy.stats as st

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'

out_file1 = '/Users/chloe/Desktop/corr1.png'

move_data = pd.read_csv(moves_file)
restart_diffoptlen = []
restart_consec_error = []
restart_consec_error_closer = []
restart_consec_error_further = []
restart_consec_error_cross = []
restart_mob = []
restart_consec_mobred = []

count_diffoptlen = []
count_consec_error = []
count_consec_error_closer = []
count_consec_error_further = []
count_consec_error_cross = []
count_mob = []
count_consec_mobred = []

hash_diffoptlen = [] # NONRESTART
hash_diffoptlen_restart = []
hash_diffoptlen_all = []
hash_consec_error = []
hash_consec_error_restart = []
hash_consec_error_all = []
hash_consec_error_closer = []
hash_consec_error_closer_restart = []
hash_consec_error_further = []
hash_consec_error_further_restart = []
hash_consec_error_cross = []
hash_consec_error_cross_restart = []
hash_mob = []
hash_mob_restart = []
hash_mob_all = []
hash_consec_mobred = []
hash_consec_mobred_restart = []

max_diffoptlen = ''
min_diffoptlen = ''
range_diffoptlen = ''
max_consec_error = ''
max_consec_error_closer = ''
max_consec_error_further = ''
max_consec_error_cross = ''
max_mob = ''
max_consec_mobred = ''


#################################### PROCESS DATA ###############################

for i in range(len(move_data)):
	row = move_data.loc[i, :]
	hash_diffoptlen_all.append(int(row['diffoptlen']))
	hash_consec_error_all.append(int(row['consec_error']))
	hash_mob_all.append(int(row['mobility']))
	'''
	# first line
	if i == 0: 
		range_diffoptlen = int(row['range_diffoptlen'])
		min_diffoptlen = int(row['min_diffoptlen'])
		max_diffoptlen = int(row['max_diffoptlen'])
		max_consec_error = int(row['max_consec_error'])
		max_consec_error_closer = int(row['max_consec_error_closer'])
		max_consec_error_further = int(row['max_consec_error_further'])
		max_consec_error_cross = int(row['max_consec_error_cross'])
		max_mob = int(row['max_mobility'])
		max_consec_mobred = int(row['max_consec_mobility_reduced'])
		restart_diffoptlen = [0] * (range_diffoptlen + 1)
		count_diffoptlen = [0] * (range_diffoptlen + 1)
		restart_consec_error = [0] * (max_consec_error + 1)
		count_consec_error = [0] * (max_consec_error + 1)
		restart_consec_error_closer = [0] * (max_consec_error_closer + 1)
		count_consec_error_closer = [0] * (max_consec_error_closer + 1)
		restart_consec_error_further = [0] * (max_consec_error_further + 1)
		count_consec_error_further = [0] * (max_consec_error_further + 1)
		restart_consec_error_cross = [0] * (max_consec_error_cross + 1)
		count_consec_error_cross = [0] * (max_consec_error_cross + 1)
		restart_mob = [0]* (max_mob + 1)
		count_mob = [0] * (max_mob + 1)
		restart_consec_mobred = [0] * (max_consec_mobred + 1)
		count_consec_mobred = [0] * (max_consec_mobred + 1)
	diffoptlen = row['diffoptlen']
	restart = row['restart']
	consec_error = row['consec_error']
	consec_error_closer = row['consec_error_closer']
	consec_error_further = row['consec_error_further']
	consec_error_cross = row['consec_error_cross']
	mobility = row['mobility']
	consec_mobred = row['consec_mobility_reduced']
	count_diffoptlen[abs(min_diffoptlen) + diffoptlen] += 1
	count_consec_error[consec_error] += 1
	count_consec_error_closer[consec_error_closer] += 1
	count_consec_error_further[consec_error_further] += 1
	count_consec_error_cross[consec_error_cross] += 1
	count_mob[mobility] += 1
	count_consec_mobred[consec_mobred] += 1
	if restart == 1:
		restart_diffoptlen[abs(min_diffoptlen) + diffoptlen] += 1
		restart_consec_error[consec_error] += 1
		restart_consec_error_closer[consec_error_closer] += 1
		restart_consec_error_further[consec_error_further] += 1
		restart_consec_error_cross[consec_error_cross] += 1
		restart_mob[mobility] += 1
		restart_consec_mobred[consec_mobred] += 1
		hash_diffoptlen_restart.append(int(diffoptlen))
		hash_consec_error_restart.append(int(consec_error))
		hash_consec_error_closer_restart.append(int(consec_error_closer))
		hash_consec_error_further_restart.append(int(consec_error_further))
		hash_consec_error_cross_restart.append(int(consec_error_cross))
		hash_mob_restart.append(int(mobility))
		hash_consec_mobred_restart.append(int(consec_mobred))
	else:
		hash_diffoptlen.append(int(diffoptlen))
		hash_consec_error.append(int(consec_error))
		hash_consec_error_closer.append(int(consec_error_closer))
		hash_consec_error_further.append(int(consec_error_further))
		hash_consec_error_cross.append(int(consec_error_cross))
		hash_mob.append(int(mobility))
		hash_consec_mobred.append(int(consec_mobred))
	'''

####################################### PLOTTING ##################################
corr_data = np.transpose(np.array([hash_diffoptlen_all, hash_mob_all, hash_consec_error_all]))
corr, p = st.spearmanr(corr_data)
print(corr)
print(p)

f, ax = plt.subplots()
sns.heatmap(corr, annot=True, fmt="f", linewidths=.5, ax=ax, cmap=sns.color_palette("Blues"))
plt.xticks([0, 1, 2], ['DiffLen', 'Mobility', 'ConsecError'])
plt.yticks([0, 1, 2], ['DiffLen', 'Mobility', 'ConsecError'])
plt.title('Spearman Correlation')
plt.savefig(out_file1)









