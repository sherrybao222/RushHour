# PLOTTING: characterize move data
# probability of making error as a function of mobility
# mobility raw value, mobility_reduced, consec_mobility_reduced
# density histograms of mobility, mobility_reduced, consec_mobility_reduced
# need to run with python27
import sys, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
np.seterr(divide='ignore', invalid='ignore')

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'

out_file10 = '/Users/chloe/Desktop/prob_error_density_mobility.png'
out_file11 = '/Users/chloe/Desktop/prob_error_density_mobred.png'
out_file12 = '/Users/chloe/Desktop/prob_error_density_consecmobred.png'

move_data = pd.read_csv(moves_file)

error_mob = []
error_mobred = []
error_consec_mobred = []

count_mob = []
count_mobred = []
count_consec_mobred = []

hash_mob = []
hash_mob_error = []
hash_mobred = []
hash_mobred_error = []
hash_consec_mobred = []
hash_consec_mobred_error = []

max_mob = ''
max_mobred = ''
max_consec_mobred = ''


#################################### PROCESS DATA ###############################

for i in range(len(move_data)):
	row = move_data.loc[i, :]
	# first line
	if i == 0: 
		max_mob = int(row['max_mobility'])
		max_mobred = 2
		max_consec_mobred = int(row['max_consec_mobility_reduced'])
		error_mob = [0] * (max_mob + 1)
		count_mob = [0] * (max_mob + 1)
		error_mobred = [0] * (max_mobred + 1)
		count_mobred = [0] * (max_mobred + 1)
		error_consec_mobred = [0] * (max_consec_mobred + 1)
		count_consec_mobred = [0] * (max_consec_mobred + 1)
	
	error = row['error']
	mobility = row['mobility']
	mobred = row['mobility_reduced']
	consec_mobred = row['consec_mobility_reduced']
	
	count_mob[mobility] += 1
	count_mobred[mobred] += 1
	count_consec_mobred[consec_mobred] += 1

	hash_mob.append(int(mobility))
	hash_mobred.append(int(mobred))
	hash_consec_mobred.append(int(consec_mobred))

	if error == 1:
		error_mob[mobility] += 1
		error_mobred[mobred] += 1
		error_consec_mobred[consec_mobred] += 1
		hash_mob_error.append(int(mobility))
		hash_mobred_error.append(int(mobred))
		hash_consec_mobred_error.append(int(consec_mobred))


####################################### PLOTTING ##################################

# mobility: prob error and density histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(len(error_mob)), \
		height=np.array(error_mob, dtype=np.float32) / np.array(count_mob, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Error', align='center')
ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_ylabel('Probability Error', fontsize=18)

ax[1].hist(hash_mob, bins=np.arange(len(error_mob))-0.5,\
			density=True, align='mid', label='Mobility', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_mob_error, bins=np.arange(len(error_mob))-0.5,\
			density=True, align='mid', label='Mobility Error', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_ylabel('Density', fontsize=18)

fig.text(0.5, 0.03, 'Mobility', ha='center', fontsize=18)
plt.suptitle('Probability of Error by and Density of Mobility', fontsize=20, weight='bold')
plt.savefig(out_file10)
plt.close()


# mobility_reduced: prob error and density histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(len(error_mobred)), \
		height=np.array(error_mobred, dtype=np.float32) / np.array(count_mobred, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Error', align='center')
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_ylabel('Probability Error', fontsize=18)

ax[1].hist(hash_mobred, bins=np.arange(len(error_mobred))-0.5,\
			density=True, align='mid', label='Mobility Reduced', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_mobred_error, bins=np.arange(len(error_mobred))-0.5,\
			density=True, align='mid', label='Mobility Reduced Error', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_ylabel('Density', fontsize=18)

fig.text(0.5, 0.03, 'Mobility Reduced', ha='center', fontsize=18)
plt.suptitle('Probability of Error by and Density of Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file11)
plt.close()

# consec_mobility_reduced: prob error and density histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(len(error_consec_mobred)), \
		height=np.array(error_consec_mobred, dtype=np.float32) / np.array(count_consec_mobred, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Error', align='center')
# ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_ylabel('Probability Error', fontsize=18)

ax[1].hist(hash_consec_mobred, bins=np.arange(len(error_consec_mobred))-0.5,\
			density=True, align='mid', label='Consecutive Mobility Reduced', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_consec_mobred_error, bins=np.arange(len(error_consec_mobred))-0.5,\
			density=True, align='mid', label='Consecutive Mobility Reduced Error', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(3))
ax[1].legend(loc=2, bbox_to_anchor=(0.35,0.9))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_ylabel('Density', fontsize=18)

fig.text(0.5, 0.03, 'Consecutive Mobility Reduced', ha='center', fontsize=18)
plt.suptitle('Probability of Error by and Density of Consecutive Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file12)
plt.close()
