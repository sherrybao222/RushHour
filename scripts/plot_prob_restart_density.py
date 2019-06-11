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

all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_optlen = [7] * len(all_instances)
ins_optlen[18:36]=[11]*len(ins_optlen[18:36])
ins_optlen[36:53]=[14]*len(ins_optlen[36:53])
ins_optlen[53:]=[16]*len(ins_optlen[53:])
optlen_consider = 14

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'

out_file21 = '/Users/chloe/Desktop/difoptlen_to_restart14.png'
out_file22 = '/Users/chloe/Desktop/mobility_to_restart14.png'
out_file23 = '/Users/chloe/Desktop/consecmobred_to_restart14.png'
out_file24 = '/Users/chloe/Desktop/consecerror_to_restart14.png'
out_file25 = '/Users/chloe/Desktop/consecerrorfurther_to_restart14.png'
out_file26 = '/Users/chloe/Desktop/consecerrorcloser_to_restart14.png'
out_file27 = '/Users/chloe/Desktop/consecerrorcross_to_restart14.png'

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
hash_consec_error = []
hash_consec_error_restart = []
hash_consec_error_closer = []
hash_consec_error_closer_restart = []
hash_consec_error_further = []
hash_consec_error_further_restart = []
hash_consec_error_cross = []
hash_consec_error_cross_restart = []
hash_mob = []
hash_mob_restart = []
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
	cur_ins = row['instance']
	if ins_optlen[all_instances.index(cur_ins)] != optlen_consider: # only consider level-16 puzzles
		continue 
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

####################################### PLOTTING ##################################
# error bar preparation (95%CI)
z = st.norm.ppf(1-0.05/2)


# Consecutive Error Cross: prob restart and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_consec_error_cross, bins=np.arange(len(count_consec_error_cross))-0.5,\
			density=True, align='mid', label='Non-restart', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error_cross), \
			color='gray', linestyle='dashed', linewidth=2.5)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_cross_restart, bins=np.arange(len(count_consec_error_cross))-0.5,\
			density=True, align='mid', label='Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_cross_restart), \
		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9), prop={'size': 16})
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.81), prop={'size': 16})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax12.locator_params(nbins=5, axis='y')
ax12.tick_params(axis='both', which='major', labelsize=16, colors='orangered')
ax[0].set_ylabel('Count', fontsize=18)
print('total consec_error_cross sample size: ', len(hash_consec_error_cross))
print('total consec_error_cross_restart sample size: ', len(hash_consec_error_cross_restart))

cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_consec_error_cross), vmax=max(count_consec_error_cross))
colors = [cmap(normalize(value)) for value in count_consec_error_cross]
count_consec_error_cross = np.array(count_consec_error_cross, dtype=np.float32)
restart_consec_error_cross = np.array(restart_consec_error_cross, dtype=np.float32)
CIup = (restart_consec_error_cross+z**2/2.0)/(count_consec_error_cross+z**2)+ (z/(count_consec_error_cross+z**2))*np.sqrt(restart_consec_error_cross*(count_consec_error_cross-restart_consec_error_cross)/count_consec_error_cross+z**2/4.0)
CIlow = (restart_consec_error_cross+z**2/2.0)/(count_consec_error_cross+z**2) - (z/(count_consec_error_cross+z**2))*np.sqrt(restart_consec_error_cross*(count_consec_error_cross-restart_consec_error_cross)/count_consec_error_cross+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error_cross)), \
		height=restart_consec_error_cross/count_consec_error_cross, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Restart', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error_cross)), restart_consec_error_cross/count_consec_error_cross, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(restart_consec_error_cross/count_consec_error_cross)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Restart', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Error Cross', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Restart; Histogram of Consecutive Error Cross', \
	fontsize=20, weight='bold')
plt.savefig(out_file27)
plt.close()

# hypothesis testing
print('\nRestart: Consecutive Error Cross')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error_cross, hash_consec_error_cross_restart,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error_cross, hash_consec_error_cross_restart,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error_cross))
print('Median Restart: ', np.median(hash_consec_error_cross_restart))
print('Mean Non-restart: ', np.mean(hash_consec_error_cross))
print('Mean Restart: ', np.mean(hash_consec_error_cross_restart))





# Consecutive Error Closer: prob restart and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_consec_error_closer, bins=np.arange(len(count_consec_error_closer))-0.5,\
			density=False, align='mid', label='Non-restart (left axis)', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error_closer), color='gray', linestyle='dashed', linewidth=1)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_closer_restart, bins=np.arange(len(count_consec_error_closer))-0.5,\
			density=False, align='mid', label='Restart (right axis)', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_closer_restart), color='orangered', linestyle='dashed', linewidth=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.83))
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax[0].set_ylabel('Count', fontsize=14)

cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_consec_error_closer), vmax=max(count_consec_error_closer))
colors = [cmap(normalize(value)) for value in count_consec_error_closer]
count_consec_error_closer = np.array(count_consec_error_closer, dtype=np.float32)
restart_consec_error_closer = np.array(restart_consec_error_closer, dtype=np.float32)
CIup = (restart_consec_error_closer+z**2/2.0)/(count_consec_error_closer+z**2)+ (z/(count_consec_error_closer+z**2))*np.sqrt(restart_consec_error_closer*(count_consec_error_closer-restart_consec_error_closer)/count_consec_error_closer+z**2/4.0)
CIlow = (restart_consec_error_closer+z**2/2.0)/(count_consec_error_closer+z**2) - (z/(count_consec_error_closer+z**2))*np.sqrt(restart_consec_error_closer*(count_consec_error_closer-restart_consec_error_closer)/count_consec_error_closer+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error_closer)), \
		height=restart_consec_error_closer/count_consec_error_closer, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Restart', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error_closer)), restart_consec_error_closer/count_consec_error_closer, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(restart_consec_error_closer/count_consec_error_closer)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Restart', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Error Closer', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Restart; Histogram of Consecutive Error Closer', \
	fontsize=20, weight='bold')
plt.savefig(out_file26)
plt.close()

# hypothesis testing
print('\nRestart: Consecutive Error Closer')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error_closer, hash_consec_error_closer_restart,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error_closer, hash_consec_error_closer_restart,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error_closer))
print('Median Restart: ', np.median(hash_consec_error_closer_restart))
print('Mean Non-restart: ', np.mean(hash_consec_error_closer))
print('Mean Restart: ', np.mean(hash_consec_error_closer_restart))






# Consecutive Error Further: prob restart and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_consec_error_further, bins=np.arange(len(count_consec_error_further))-0.5,\
			density=False, align='mid', label='Non-restart (left axis)', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error_further), color='gray', linestyle='dashed', linewidth=1)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_further_restart, bins=np.arange(len(count_consec_error_further))-0.5,\
			density=False, align='mid', label='Restart (right axis)', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_further_restart), color='orangered', linestyle='dashed', linewidth=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.83))
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax[0].set_ylabel('Count', fontsize=14)

cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_consec_error_further), vmax=max(count_consec_error_further))
colors = [cmap(normalize(value)) for value in count_consec_error_further]
count_consec_error_further = np.array(count_consec_error_further, dtype=np.float32)
restart_consec_error_further = np.array(restart_consec_error_further, dtype=np.float32)
CIup = (restart_consec_error_further+z**2/2.0)/(count_consec_error_further+z**2)+ (z/(count_consec_error_further+z**2))*np.sqrt(restart_consec_error_further*(count_consec_error_further-restart_consec_error_further)/count_consec_error_further+z**2/4.0)
CIlow = (restart_consec_error_further+z**2/2.0)/(count_consec_error_further+z**2) - (z/(count_consec_error_further+z**2))*np.sqrt(restart_consec_error_further*(count_consec_error_further-restart_consec_error_further)/count_consec_error_further+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error_further)), \
		height=restart_consec_error_further/count_consec_error_further, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Restart', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error_further)), restart_consec_error_further/count_consec_error_further, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(restart_consec_error_further/count_consec_error_further)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Restart', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Error Further', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Restart; Histogram of Consecutive Error Further', \
	fontsize=20, weight='bold')
plt.savefig(out_file25)
plt.close()

# hypothesis testing
print('\nRestart: Consecutive Error Further')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error_further, hash_consec_error_further_restart,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error_further, hash_consec_error_further_restart,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error_further))
print('Median Restart: ', np.median(hash_consec_error_further_restart))
print('Mean Non-restart: ', np.mean(hash_consec_error_further))
print('Mean Restart: ', np.mean(hash_consec_error_further_restart))






# Consecutive Error: prob restart and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_consec_error, bins=np.arange(len(count_consec_error))-0.5,\
			density=True, align='mid', label='Non-restart', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error), \
			color='gray', linestyle='dashed', linewidth=2.5)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_restart, bins=np.arange(len(count_consec_error))-0.5,\
			density=True, align='mid', label='Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_restart), \
		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9), prop={'size': 16})
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.81), prop={'size': 16})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax12.locator_params(nbins=5, axis='y')
ax12.tick_params(axis='both', which='major', labelsize=16, colors='orangered')
ax[0].set_ylabel('Count', fontsize=18)
print('total consec_error sample size: ', len(hash_consec_error))
print('total consec_error_restart sample size: ', len(hash_consec_error_restart))

cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_consec_error), vmax=max(count_consec_error))
colors = [cmap(normalize(value)) for value in count_consec_error]
count_consec_error = np.array(count_consec_error, dtype=np.float32)
restart_consec_error = np.array(restart_consec_error, dtype=np.float32)
CIup = (restart_consec_error+z**2/2.0)/(count_consec_error+z**2)+ (z/(count_consec_error+z**2))*np.sqrt(restart_consec_error*(count_consec_error-restart_consec_error)/count_consec_error+z**2/4.0)
CIlow = (restart_consec_error+z**2/2.0)/(count_consec_error+z**2) - (z/(count_consec_error+z**2))*np.sqrt(restart_consec_error*(count_consec_error-restart_consec_error)/count_consec_error+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error)), \
		height=restart_consec_error/count_consec_error, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Restart', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error)), restart_consec_error/count_consec_error, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(restart_consec_error/count_consec_error)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Restart', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Error', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Restart; Histogram of Consecutive Error', \
	fontsize=20, weight='bold')
plt.savefig(out_file24)
plt.close()

# hypothesis testing
print('\nRestart: Consecutive Error')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error, hash_consec_error_restart,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error, hash_consec_error_restart,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error))
print('Median Restart: ', np.median(hash_consec_error_restart))
print('Mean Non-restart: ', np.mean(hash_consec_error))
print('Mean Restart: ', np.mean(hash_consec_error_restart))







# Consecutive Mobility Reduced: prob restart and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_consec_mobred, bins=np.arange(len(count_consec_mobred))-0.5,\
			density=False, align='mid', label='Non-restart (left axis)', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_mobred), color='gray', linestyle='dashed', linewidth=1)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_mobred_restart, bins=np.arange(len(count_consec_mobred))-0.5,\
			density=False, align='mid', label='Restart (right axis)', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_mobred_restart), color='orangered', linestyle='dashed', linewidth=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.83))
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax[0].set_ylabel('Count', fontsize=14)

cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_consec_mobred), vmax=max(count_consec_mobred))
colors = [cmap(normalize(value)) for value in count_consec_mobred]
count_consec_mobred = np.array(count_consec_mobred, dtype=np.float32)
restart_consec_mobred = np.array(restart_consec_mobred, dtype=np.float32)
CIup = (restart_consec_mobred+z**2/2.0)/(count_consec_mobred+z**2)+ (z/(count_consec_mobred+z**2))*np.sqrt(restart_consec_mobred*(count_consec_mobred-restart_consec_mobred)/count_consec_mobred+z**2/4.0)
CIlow = (restart_consec_mobred+z**2/2.0)/(count_consec_mobred+z**2) - (z/(count_consec_mobred+z**2))*np.sqrt(restart_consec_mobred*(count_consec_mobred-restart_consec_mobred)/count_consec_mobred+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_mobred)), \
		height=restart_consec_mobred/count_consec_mobred, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Restart', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_mobred)), restart_consec_mobred/count_consec_mobred, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(restart_consec_mobred/count_consec_mobred)+0.02)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Restart', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Mobility Reduced', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Restart; Histogram of Consecutive Mobility Reduced', \
	fontsize=20, weight='bold')
plt.savefig(out_file23)
plt.close()

# hypothesis testing
print('\nRestart: Consecutive Mobility Reduced')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_mobred, hash_consec_mobred_restart,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_mobred, hash_consec_mobred_restart,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_mobred))
print('Median Restart: ', np.median(hash_consec_mobred_restart))
print('Mean Non-restart: ', np.mean(hash_consec_mobred))
print('Mean Restart: ', np.mean(hash_consec_mobred_restart))









# Mobility: prob restart and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_mob, bins=np.arange(len(count_mob))-0.5,\
			density=True, align='mid', label='Non-restart', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_mob), \
			color='gray', linestyle='dashed', linewidth=2.5)
ax12 = ax[0].twinx()
ax12.hist(hash_mob_restart, bins=np.arange(len(count_mob))-0.5,\
			density=True, align='mid', label='Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax12.axvline(np.median(hash_mob_restart), \
		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9), prop={'size': 16})
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.81), prop={'size': 16})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax12.locator_params(nbins=5, axis='y')
ax12.tick_params(axis='both', which='major', labelsize=16, colors='orangered')
ax[0].set_ylabel('Count', fontsize=18)
print('total mob sample size: ', len(hash_mob))
print('total mob_restart sample size: ', len(hash_mob_restart))


cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_mob), vmax=max(count_mob))
colors = [cmap(normalize(value)) for value in count_mob]
count_mob = np.array(count_mob, dtype=np.float32)
restart_mob = np.array(restart_mob, dtype=np.float32)
CIup = (restart_mob+z**2/2.0)/(count_mob+z**2)+ (z/(count_mob+z**2))*np.sqrt(restart_mob*(count_mob-restart_mob)/count_mob+z**2/4.0)
CIlow = (restart_mob+z**2/2.0)/(count_mob+z**2) - (z/(count_mob+z**2))*np.sqrt(restart_mob*(count_mob-restart_mob)/count_mob+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_mob)), \
		height=restart_mob/count_mob, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Restart', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_mob)), restart_mob/count_mob, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(restart_mob/count_mob)+0.015)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Restart', fontsize=14)

fig.text(0.5, 0.029, \
	'Mobility', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Restart; Histogram of Mobility', \
	fontsize=20, weight='bold')
plt.savefig(out_file22)
plt.close()

# hypothesis testing
print('\nRestart: Mobility')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_mob, hash_mob_restart,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_mob, hash_mob_restart,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_mob))
print('Median Restart: ', np.median(hash_mob_restart))
print('Mean Non-restart: ', np.mean(hash_mob))
print('Mean Restart: ', np.mean(hash_mob_restart))






# diffoptlen: prob restart and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_diffoptlen, bins=np.arange(start=min_diffoptlen,stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Non-restart', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_diffoptlen), \
			color='gray', linestyle='dashed', linewidth=2.5)
ax12 = ax[0].twinx()
ax12.hist(hash_diffoptlen_restart, bins=np.arange(start=min_diffoptlen,stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax12.axvline(np.median(hash_diffoptlen_restart), \
		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9), prop={'size': 16})
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.81), prop={'size': 16})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax12.locator_params(nbins=5, axis='y')
ax12.tick_params(axis='both', which='major', labelsize=16, colors='orangered')
ax[0].set_ylabel('Count', fontsize=18)
print('total diffoptlen sample size: ', len(hash_diffoptlen))
print('total diffoptlen_restart sample size: ', len(hash_diffoptlen_restart))

cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_diffoptlen), vmax=max(count_diffoptlen))
colors = [cmap(normalize(value)) for value in count_diffoptlen]
count_diffoptlen = np.array(count_diffoptlen, dtype=np.float32)
restart_diffoptlen = np.array(restart_diffoptlen, dtype=np.float32)
CIup = (restart_diffoptlen+z**2/2.0)/(count_diffoptlen+z**2)+ (z/(count_diffoptlen+z**2))*np.sqrt(restart_diffoptlen*(count_diffoptlen-restart_diffoptlen)/count_diffoptlen+z**2/4.0)
CIlow = (restart_diffoptlen+z**2/2.0)/(count_diffoptlen+z**2) - (z/(count_diffoptlen+z**2))*np.sqrt(restart_diffoptlen*(count_diffoptlen-restart_diffoptlen)/count_diffoptlen+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_diffoptlen)), \
		height=restart_diffoptlen/count_diffoptlen, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Restart', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_diffoptlen)), restart_diffoptlen/count_diffoptlen, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(restart_diffoptlen/count_diffoptlen)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Restart', fontsize=14)

fig.text(0.5, 0.029, \
	'Difference of Current Length and Initial Optlen', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Restart; Histogram of Length Difference', \
	fontsize=20, weight='bold')
plt.savefig(out_file21)
plt.close()

# hypothesis testing
print('\nRestart: Length Difference')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_diffoptlen, hash_diffoptlen_restart,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_diffoptlen, hash_diffoptlen_restart,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_diffoptlen))
print('Median Restart: ', np.median(hash_diffoptlen_restart))
print('Mean Non-restart: ', np.mean(hash_diffoptlen))
print('Mean Restart: ', np.mean(hash_diffoptlen_restart))

















sys.exit()
##################################### OLD PLOT ##################################

# out_file1 = '/Users/chloe/Desktop/prob_restart_diffoptlen.png'
# out_file2 = '/Users/chloe/Desktop/prob_restart_consec_error.png'
# out_file3 = '/Users/chloe/Desktop/prob_restart_consec_error_closer.png'
# out_file4 = '/Users/chloe/Desktop/prob_restart_consec_error_further.png'

# out_file5 = '/Users/chloe/Desktop/hist_optlen.png'
# out_file6 = '/Users/chloe/Desktop/hist_consec_error.png'
# out_file7 = '/Users/chloe/Desktop/hist_consec_error_closer.png'
# out_file8 = '/Users/chloe/Desktop/hist_consec_error_further.png'

# out_file9 = '/Users/chloe/Desktop/prob_restart_consec_all_error.png'

# out_file10 = '/Users/chloe/Desktop/prob_restart_error.png'
# out_file11 = '/Users/chloe/Desktop/density_restart_error.png'
# out_file12 = '/Users/chloe/Desktop/prob_restart_density_diffoptlen.png'

# out_file13 = '/Users/chloe/Desktop/prob_restart_mob.png'
# out_file14 = '/Users/chloe/Desktop/density_restart_mob.png'

# out_file15 = '/Users/chloe/Desktop/count_restart_error.png'
# out_file16 = '/Users/chloe/Desktop/count_restart_mob.png'

# density of mobility and consec_mobred
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].hist(hash_mob, bins=np.arange(len(restart_mob))-0.5,\
			density=False, align='mid', label='Mobility', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Mobility', fontsize=15)
ax[0, 1].hist(hash_mob_restart, bins=np.arange(len(restart_mob))-0.5,\
			density=False, align='mid', label='Mobility Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_xlabel('Mobility Restart', fontsize=18)

ax[1, 0].hist(hash_consec_mobred, bins=np.arange(len(restart_consec_mobred))-0.5,\
			density=False, align='mid', label='Consecutive Mobility Reduced', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_xlabel('Consecutive Mobility Reduced', fontsize=18)
ax[1, 1].hist(hash_consec_mobred_restart, bins=np.arange(len(restart_consec_mobred))-0.5,\
			density=False, align='mid', label='Consecutive Mobility Reduced Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_xlabel('Consecutive Mobility Reduced Restart', fontsize=18)

fig.text(0.06, 0.5, 'Count', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Histogram of Mobility and Consecutive Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file16)
plt.close()

sys.exit()


# count histogram of consec errors with restart
fig, ax = plt.subplots(4, 2, figsize=(15, 16))
ax[0, 0].hist(hash_consec_error, bins=np.arange(len(restart_consec_error))-0.5,\
			density=False, align='mid', label='Overall', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)
ax[0, 1].hist(hash_consec_error_restart, bins=np.arange(len(restart_consec_error))-0.5,\
			density=False, align='mid', label='Overall Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_title('Overall Restart', fontsize=15)

ax[1, 0].hist(hash_consec_error_cross, bins=np.arange(len(restart_consec_error_cross))-0.5,\
			density=False, align='mid', label='Cross', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)
ax[1, 1].hist(hash_consec_error_cross_restart, bins=np.arange(len(restart_consec_error_cross))-0.5,\
			density=False, align='mid', label='Cross Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Cross Restart', fontsize=15)

ax[2, 0].hist(hash_consec_error_further, bins=np.arange(len(restart_consec_error_further))-0.5,\
			density=False, align='mid', label='Further', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[2, 0].tick_params(axis='both', which='major', labelsize=13)
ax[2, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))
ax[2, 0].set_title('Further', fontsize=15)
ax[2, 1].hist(hash_consec_error_further_restart, bins=np.arange(len(restart_consec_error_further))-0.5,\
			density=False, align='mid', label='Further Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[2, 1].tick_params(axis='both', which='major', labelsize=13)
ax[2, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
ax[2, 1].set_title('Further Restart', fontsize=15)

ax[3, 0].hist(hash_consec_error_closer, bins=np.arange(len(restart_consec_error_closer))-0.5,\
			density=False, align='mid', label='Closer', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[3, 0].tick_params(axis='both', which='major', labelsize=13)
ax[3, 0].set_title('Closer', fontsize=15)
ax[3, 1].hist(hash_consec_error_closer_restart, bins=np.arange(len(restart_consec_error_closer))-0.5,\
			density=False, align='mid', label='Closer Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[3, 1].tick_params(axis='both', which='major', labelsize=13)
ax[3, 1].set_title('Closer Restart', fontsize=15)


fig.text(0.5, 0.06, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.045, 0.5, 'Count', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Histogram of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file15)
plt.close()

# sys.exit()


# prob restart of mobility and consec_mobred
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(len(restart_mob)), \
		height=np.array(restart_mob, dtype=np.float32) / np.array(count_mob, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Restart', align='center')
# ax[0].xaxis.set_major_locator(ticker.MultipleLocator())
ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_xlabel('Mobility', fontsize=18)

ax[1].bar(x=np.arange(len(restart_consec_mobred)), \
		height=np.array(restart_consec_mobred, dtype=np.float32) / np.array(count_consec_mobred, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Restart', align='center')
# ax[1].xaxis.set_major_locator(ticker.MultipleLocator())
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_xlabel('Consecutive Mobility Reduced', fontsize=18)

fig.text(0.06, 0.5, 'Probability Restart', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Probability Restart as a Function of Mobility and Consecutive Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file13)
plt.close()


# density of mobility and consec_mobred
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].hist(hash_mob, bins=np.arange(len(restart_mob))-0.5,\
			density=True, align='mid', label='Mobility', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].hist(hash_mob_restart, bins=np.arange(len(restart_mob))-0.5,\
			density=True, align='mid', label='Mobility Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_xlabel('Mobility', fontsize=18)

ax[1].hist(hash_consec_mobred, bins=np.arange(len(restart_consec_mobred))-0.5,\
			density=True, align='mid', label='Consecutive Mobility Reduced', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_consec_mobred_restart, bins=np.arange(len(restart_consec_mobred))-0.5,\
			density=True, align='mid', label='Consecutive Mobility Reduced Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[1].legend(loc=2, bbox_to_anchor=(0.35,0.9))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_xlabel('Consecutive Mobility Reduced', fontsize=18)

fig.text(0.06, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Density of Mobility and Consecutive Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file14)
plt.close()



# diffoptlen: prob restart and density histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(start=min_diffoptlen, stop=max_diffoptlen+1), \
		height=np.array(restart_diffoptlen, dtype=np.float32) / np.array(count_diffoptlen, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Restart', align='center')
ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_ylabel('Probability Restart', fontsize=18)

ax[1].hist(hash_diffoptlen, bins=np.arange(start=min_diffoptlen,stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Diffoptlen', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_diffoptlen_restart, bins=np.arange(start=min_diffoptlen,stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Diffoptlen Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_ylabel('Density', fontsize=18)

fig.text(0.5, 0.03, 'DiffOptlen', ha='center', fontsize=18)
plt.suptitle('Probability of Restart by and Density of Length Difference', fontsize=20, weight='bold')
plt.savefig(out_file12)
plt.close()

# sys.exit()

# density of consec errors with restart
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].hist(hash_consec_error, bins=np.arange(len(restart_consec_error))-0.5,\
			density=True, align='mid', label='Overall', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 0].hist(hash_consec_error_restart, bins=np.arange(len(restart_consec_error))-0.5,\
			density=True, align='mid', label='Overall Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[0, 0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)

ax[1, 0].hist(hash_consec_error_cross, bins=np.arange(len(restart_consec_error_cross))-0.5,\
			density=True, align='mid', label='Cross', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 0].hist(hash_consec_error_cross_restart, bins=np.arange(len(restart_consec_error_cross))-0.5,\
			density=True, align='mid', label='Cross Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[1, 0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)

ax[0, 1].hist(hash_consec_error_further, bins=np.arange(len(restart_consec_error_further))-0.5,\
			density=True, align='mid', label='Further', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 1].hist(hash_consec_error_further_restart, bins=np.arange(len(restart_consec_error_further))-0.5,\
			density=True, align='mid', label='Further Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[0, 1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
ax[0, 1].set_title('Further', fontsize=15)

ax[1, 1].hist(hash_consec_error_closer, bins=np.arange(len(restart_consec_error_closer))-0.5,\
			density=True, align='mid', label='Closer', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 1].hist(hash_consec_error_closer_restart, bins=np.arange(len(restart_consec_error_closer))-0.5,\
			density=True, align='mid', label='Closer Restart', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[1, 1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Closer', fontsize=15)

fig.text(0.5, 0.04, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.07, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Density of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file11)
plt.close()


# prob restart of consec errors
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].bar(x=np.arange(len(restart_consec_error)), \
		height=np.array(restart_consec_error, dtype=np.float32) / np.array(count_consec_error, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Restart', align='center')
ax[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(8))
ax[0, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)

ax[1, 0].bar(x=np.arange(len(restart_consec_error_cross)), \
		height=np.array(restart_consec_error_cross, dtype=np.float32) / np.array(count_consec_error_cross, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Restart', align='center')
ax[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(8))
ax[1, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)

ax[0, 1].bar(x=np.arange(len(restart_consec_error_further)), \
		height=np.array(restart_consec_error_further, dtype=np.float32) / np.array(count_consec_error_further, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Restart', align='center')
ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(4))
ax[0, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_title('Further', fontsize=15)

ax[1, 1].bar(x=np.arange(len(restart_consec_error_closer)), \
		height=np.array(restart_consec_error_closer, dtype=np.float32) / np.array(count_consec_error_closer, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Restart', align='center')
ax[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(7))
ax[1, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Closer', fontsize=15)

fig.text(0.5, 0.04, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.07, 0.5, 'Probability Restart', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Probability Restart as a Function of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file10)
plt.close()








sys.exit()
##################################### OLD PLOT ##################################

# prob optlen
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax2 = fig.add_subplot(111, label='2', frame_on=False)
ax.bar(x=np.arange(len(restart_diffoptlen)), \
		height=np.array(restart_diffoptlen, dtype=np.float32) / np.array(count_diffoptlen, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=1,
		label='Probability Restart', align='center')
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Probability Restart', fontsize=14)
ax.set_xlabel('Optlen', fontsize=14)
ax2.hist(hash_diffoptlen, bins=np.arange(len(restart_diffoptlen))-0.5,\
			density=True, align='mid', label='Density Optlen', \
			color='blue', edgecolor='blue', alpha=0.15, width=1)
ax2.hist(hash_diffoptlen_restart, bins=np.arange(len(restart_diffoptlen))-0.5,\
			density=True, align='mid', label='Density Optlen Restart', \
			color='green', edgecolor='green', alpha=0.15, width=1)
ax2.yaxis.tick_right()
ax2.xaxis.set_ticks([])
ax2.set_ylabel('Density', fontsize=14)
ax2.yaxis.set_label_position('right')
plt.title('Probability of Restart as a Function of Optlen',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.55,0.8))
plt.savefig(out_file1)
plt.close()


# prob consec error
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax.bar(x=np.arange(len(restart_consec_error)), \
		height=np.array(restart_consec_error, dtype=np.float32) / np.array(count_consec_error, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=1,
		label='Probability Restart', align='center')
ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Probability Restart', fontsize=14)
ax.set_xlabel('Consecutive Error', fontsize=14)
ax2 = fig.add_subplot(111, label='2', frame_on=False)
ax2.hist(hash_consec_error, bins=np.arange(len(restart_consec_error)+1)-0.5,\
			density=True, align='mid', label='Density Consecutive Error', \
			color='blue', edgecolor='blue', alpha=0.15, width=1)
ax2.hist(hash_consec_error_restart, bins=np.arange(len(restart_consec_error)+1)-0.5,\
			density=True, align='mid', label='Density Consecutive Error Restart', \
			color='green', edgecolor='green', alpha=0.15, width=1)
ax2.yaxis.tick_right()
ax2.xaxis.set_ticks([])
ax2.set_ylabel('Density', fontsize=14)
ax2.yaxis.set_label_position('right')
plt.title('Probability of Restart as a Function of Consecutive Error',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.25,0.85))
plt.savefig(out_file2)
plt.close()



# prob consec error closer
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax2 = fig.add_subplot(111, label='2', frame_on=False)
ax.bar(x=np.arange(len(restart_consec_error_closer)), \
		height=np.array(restart_consec_error_closer, dtype=np.float32) / np.array(count_consec_error_closer, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=1,
		label='Probability Restart', align='center')
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Probability Restart', fontsize=14)
ax.set_xlabel('Consecutive Error Closer', fontsize=14)
ax2.hist(hash_consec_error_closer, bins=np.arange(len(restart_consec_error_closer)+1)-0.5,\
			density=True, align='mid', label='Density Consecutive Error Closer', \
			color='blue', edgecolor='blue', alpha=0.15, width=1)
ax2.hist(hash_consec_error_closer_restart, bins=np.arange(len(restart_consec_error_closer)+1)-0.5,\
			density=True, align='mid', label='Density Consecutive Error Closer Restart', \
			color='green', edgecolor='green', alpha=0.15, width=1)
ax2.yaxis.tick_right()
ax2.xaxis.set_ticks([])
ax2.set_ylabel('Density', fontsize=14)
ax2.yaxis.set_label_position('right')
plt.title('Probability of Restart as a Function of Consecutive Error Closer',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.19,0.85))
plt.savefig(out_file3)
plt.close()



# prob consec error further
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax2 = fig.add_subplot(111, label='2', frame_on=False)
ax.bar(x=np.arange(len(restart_consec_error_further)), \
		height=np.array(restart_consec_error_further, dtype=np.float32) / np.array(count_consec_error_further, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=1,
		label='Probability Restart', align='center')
ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Probability Restart', fontsize=14)
ax.set_xlabel('Consecutive Error Further', fontsize=14)
ax2.hist(hash_consec_error_further, bins=np.arange(len(restart_consec_error_further)+1)-0.5,\
			density=True, align='mid', label='Density Consecutive Error Further', \
			color='blue', edgecolor='blue', alpha=0.15, width=1)
ax2.hist(hash_consec_error_further_restart, bins=np.arange(len(restart_consec_error_further)+1)-0.5,\
			density=True, align='mid', label='Density Consecutive Error Further Restart', \
			color='green', edgecolor='green', alpha=0.15, width=1)
ax2.yaxis.tick_right()
ax2.xaxis.set_ticks([])
ax2.set_ylabel('Density', fontsize=14)
ax2.yaxis.set_label_position('right')
plt.title('Probability of Restart as a Function of Consecutive Error Further',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.25,0.85))
plt.savefig(out_file4)
plt.close()



# prob all error
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax.bar(x=np.arange(len(restart_consec_error_further)), \
		height=np.array(restart_consec_error_further, dtype=np.float32) / np.array(count_consec_error_further, dtype=np.float32), \
		width=1, color='red', edgecolor='red', alpha=1,
		align='center',\
		label='Consecutive Error Further')
ax.bar(x=np.arange(len(restart_consec_error)), \
		height=np.array(restart_consec_error, dtype=np.float32) / np.array(count_consec_error, dtype=np.float32), \
		width=1, color='blue', edgecolor='blue', alpha=1,
		align='center',\
		label='Consecutive Error')
ax.bar(x=np.arange(len(restart_consec_error_closer)), \
		height=np.array(restart_consec_error_closer, dtype=np.float32) / np.array(count_consec_error_closer, dtype=np.float32), \
		width=1, color='green', edgecolor='green', alpha=0.6,
		align='center',\
		label='Consecutive Error Closer')
ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Probability Restart', fontsize=14)
ax.set_xlabel('Error', fontsize=14)

plt.title('Probability of Restart as a Function of All Types of Errors',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.25,0.85))
plt.savefig(out_file9)
plt.close()








# hist optlen
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax.hist(hash_diffoptlen, bins=np.arange(len(restart_diffoptlen))-0.5,\
			density=False, align='mid', label='Count Optlen', \
			color='gray', edgecolor='gray', alpha=0.9, width=1)
ax.hist(hash_diffoptlen_restart, bins=np.arange(len(restart_diffoptlen))-0.5,\
			density=False, align='mid', label='Count Optlen Restart', \
			color='black', edgecolor='black', alpha=0.6, width=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Count', fontsize=14)
ax.set_xlabel('Optlen', fontsize=14)
plt.title('Histogram of Optlen',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.55,0.8))
plt.savefig(out_file5)
plt.close()



# hist consec error
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax.hist(hash_consec_error, bins=np.arange(len(restart_consec_error)+1)-0.5,\
			density=False, align='mid', label='Count Consecutive Error', \
			color='gray', edgecolor='gray', alpha=0.9, width=1)
ax.hist(hash_consec_error_restart, bins=np.arange(len(restart_consec_error)+1)-0.5,\
			density=False, align='mid', label='Count Consecutive Error Restart', \
			color='black', edgecolor='black', alpha=0.6, width=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Count', fontsize=14)
ax.set_xlabel('Consecutive Error', fontsize=14)
plt.title('Histogram of Consecutive Error',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.25,0.85))
plt.savefig(out_file6)
plt.close()


# hist consec error closer
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax.hist(hash_consec_error_closer, bins=np.arange(len(restart_consec_error_closer)+1)-0.5,\
			density=False, align='mid', label='Count Consecutive Error Closer', \
			color='gray', edgecolor='gray', alpha=0.9, width=1)
ax.hist(hash_consec_error_closer_restart, bins=np.arange(len(restart_consec_error_closer)+1)-0.5,\
			density=False, align='mid', label='Count Consecutive Error Closer Restart', \
			color='black', edgecolor='black', alpha=0.6, width=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Count', fontsize=14)
ax.set_xlabel('Consecutive Error Closer', fontsize=14)
plt.title('Histogram of Consecutive Error Closer',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.2,0.85))
plt.savefig(out_file7)
plt.close()



# hist consec error further
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111, label='1')
ax.hist(hash_consec_error_further, bins=np.arange(len(restart_consec_error_further)+1)-0.5,\
			density=False, align='mid', label='Count Consecutive Error Further', \
			color='gray', edgecolor='gray', alpha=0.9, width=1)
ax.hist(hash_consec_error_further_restart, bins=np.arange(len(restart_consec_error_further)+1)-0.5,\
			density=False, align='mid', label='Count Consecutive Error Further Restart', \
			color='black', edgecolor='black', alpha=0.6, width=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_ylabel('Count', fontsize=14)
ax.set_xlabel('Consecutive Error Further', fontsize=14)
plt.title('Histogram of Consecutive Error Further',\
			weight='bold', fontsize=15)
fig.legend(loc=2, bbox_to_anchor=(0.25,0.85))
plt.savefig(out_file8)
plt.close()





















