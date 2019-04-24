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

out_file10 = '/Users/chloe/Desktop/prob_mobred_error.png'
out_file11 = '/Users/chloe/Desktop/density_mobred_error.png'
out_file12 = '/Users/chloe/Desktop/prob_density_mobred_error.png'
move_data = pd.read_csv(moves_file)

mobred_error = []
mobred_consec_error = []
mobred_consec_error_closer = []
mobred_consec_error_further = []
mobred_consec_error_cross = []

count_error = []
count_consec_error = []
count_consec_error_closer = []
count_consec_error_further = []
count_consec_error_cross = []

hash_error = []
hash_error_mobred = []
hash_consec_error = []
hash_consec_error_mobred = []
hash_consec_error_closer = []
hash_consec_error_closer_mobred = []
hash_consec_error_further = []
hash_consec_error_further_mobred = []
hash_consec_error_cross = []
hash_consec_error_cross_mobred = []

max_error = ''
max_consec_error = ''
max_consec_error_closer = ''
max_consec_error_further = ''
max_consec_error_cross = ''


#################################### PROCESS DATA ###############################

for i in range(len(move_data)):
	row = move_data.loc[i, :]
	# first line
	if i == 0: 
		max_error = 2
		max_consec_error = int(row['max_consec_error'])
		max_consec_error_closer = int(row['max_consec_error_closer'])
		max_consec_error_further = int(row['max_consec_error_further'])
		max_consec_error_cross = int(row['max_consec_error_cross'])

		mobred_error = [0] * (max_error + 1)
		count_error = [0] * (max_error + 1)
		mobred_consec_error = [0] * (max_consec_error + 1)
		count_consec_error = [0] * (max_consec_error + 1)
		mobred_consec_error_closer = [0] * (max_consec_error_closer + 1)
		count_consec_error_closer = [0] * (max_consec_error_closer + 1)
		mobred_consec_error_further = [0] * (max_consec_error_further + 1)
		count_consec_error_further = [0] * (max_consec_error_further + 1)
		mobred_consec_error_cross = [0] * (max_consec_error_cross + 1)
		count_consec_error_cross = [0] * (max_consec_error_cross + 1)

	error = row['error']
	mobred = row['mobility_reduced']
	consec_error = row['consec_error']
	consec_error_closer = row['consec_error_closer']
	consec_error_further = row['consec_error_further']
	consec_error_cross = row['consec_error_cross']

	count_error[error] += 1
	count_consec_error[consec_error] += 1
	count_consec_error_closer[consec_error_closer] += 1
	count_consec_error_further[consec_error_further] += 1
	count_consec_error_cross[consec_error_cross] += 1

	hash_error.append(int(error))
	hash_consec_error.append(int(consec_error))
	hash_consec_error_closer.append(int(consec_error_closer))
	hash_consec_error_further.append(int(consec_error_further))
	hash_consec_error_cross.append(int(consec_error_cross))

	if mobred == 1:
		mobred_error[error] += 1
		mobred_consec_error[consec_error] += 1
		mobred_consec_error_closer[consec_error_closer] += 1
		mobred_consec_error_further[consec_error_further] += 1
		mobred_consec_error_cross[consec_error_cross] += 1
		hash_error_mobred.append(int(error))
		hash_consec_error_mobred.append(int(consec_error))
		hash_consec_error_closer_mobred.append(int(consec_error_closer))
		hash_consec_error_further_mobred.append(int(consec_error_further))
		hash_consec_error_cross_mobred.append(int(consec_error_cross))


####################################### PLOTTING ##################################

# prob mobred of consec errors
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].bar(x=np.arange(len(mobred_consec_error)), \
		height=np.array(mobred_consec_error, dtype=np.float32) / np.array(count_consec_error, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Mobility Reduced', align='center')
ax[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(8))
ax[0, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)

ax[1, 0].bar(x=np.arange(len(mobred_consec_error_cross)), \
		height=np.array(mobred_consec_error_cross, dtype=np.float32) / np.array(count_consec_error_cross, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Mobility Reduced', align='center')
ax[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(8))
ax[1, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)

ax[0, 1].bar(x=np.arange(len(mobred_consec_error_further)), \
		height=np.array(mobred_consec_error_further, dtype=np.float32) / np.array(count_consec_error_further, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Mobility Reduced', align='center')
ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(6))
ax[0, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_title('Further', fontsize=15)

ax[1, 1].bar(x=np.arange(len(mobred_consec_error_closer)), \
		height=np.array(mobred_consec_error_closer, dtype=np.float32) / np.array(count_consec_error_closer, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Mobility Reduced', align='center')
ax[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(7))
ax[1, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Closer', fontsize=15)

fig.text(0.5, 0.04, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.07, 0.5, 'Probability Mobility Reduced', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Probability of Mobility Reduced as a Function of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file10)
plt.close()

# density of consec errors with mobred
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].hist(hash_consec_error, bins=np.arange(len(mobred_consec_error))-0.5,\
			density=True, align='mid', label='Overall', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 0].hist(hash_consec_error_mobred, bins=np.arange(len(mobred_consec_error))-0.5,\
			density=True, align='mid', label='Overall Mobility Reduced', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[0, 0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)

ax[1, 0].hist(hash_consec_error_cross, bins=np.arange(len(mobred_consec_error_cross))-0.5,\
			density=True, align='mid', label='Cross', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 0].hist(hash_consec_error_cross_mobred, bins=np.arange(len(mobred_consec_error_cross))-0.5,\
			density=True, align='mid', label='Cross Mobility Reduced', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[1, 0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)

ax[0, 1].hist(hash_consec_error_further, bins=np.arange(len(mobred_consec_error_further))-0.5,\
			density=True, align='mid', label='Further', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 1].hist(hash_consec_error_further_mobred, bins=np.arange(len(mobred_consec_error_further))-0.5,\
			density=True, align='mid', label='Further Mobility Reduced', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[0, 1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_title('Further', fontsize=15)

ax[1, 1].hist(hash_consec_error_closer, bins=np.arange(len(mobred_consec_error_closer))-0.5,\
			density=True, align='mid', label='Closer', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 1].hist(hash_consec_error_closer_mobred, bins=np.arange(len(mobred_consec_error_closer))-0.5,\
			density=True, align='mid', label='Closer Mobility Reduced', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[1, 1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Closer', fontsize=15)

fig.text(0.5, 0.04, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.07, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Density of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file11)
plt.close()


# error(0,1): prob mobred and density histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(len(mobred_error)), \
		height=np.array(mobred_error, dtype=np.float32) / np.array(count_error, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Mobility Reduced', align='center')
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_ylabel('Probability Mobility Reduced', fontsize=18)
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

ax[1].hist(hash_error, bins=np.arange(len(mobred_error))-0.5,\
			density=True, align='mid', label='Error', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_error_mobred, bins=np.arange(len(mobred_error))-0.5,\
			density=True, align='mid', label='Error Mobility Reduced', \
			color='teal', edgecolor='teal', alpha=0.3, width=1)
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[1].legend(loc=2, bbox_to_anchor=(0.5,0.7))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_ylabel('Density', fontsize=18)

fig.text(0.5, 0.03, 'Error', ha='center', fontsize=18)
plt.suptitle('Probability of Mobility Reduced by and Density of Error', fontsize=20, weight='bold')
plt.savefig(out_file12)
plt.close()





