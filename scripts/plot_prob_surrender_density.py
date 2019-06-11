# PLOTTING: characterize move data
# probability of surrender as a function of optlen or consec errors
# density histograms of optlen or consec errors
# need to run with python27
import sys, csv
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

out_file21 = '/Users/chloe/Desktop/difoptlen_to_surrender14.png'
out_file22 = '/Users/chloe/Desktop/mobility_to_surrender14.png'
out_file23 = '/Users/chloe/Desktop/consecmobred_to_surrender14.png'
out_file24 = '/Users/chloe/Desktop/consecerror_to_surrender14.png'
out_file25 = '/Users/chloe/Desktop/consecerrorfurther_to_surrender14.png'
out_file26 = '/Users/chloe/Desktop/consecerrorcloser_to_surrender14.png'
out_file27 = '/Users/chloe/Desktop/consecerrorcross_to_surrender14.png'


move_data = pd.read_csv(moves_file)

surrender_diffoptlen = []
surrender_consec_error = []
surrender_consec_error_closer = []
surrender_consec_error_further = []
surrender_consec_error_cross = []
surrender_mob = []
surrender_consec_mobred = []

count_diffoptlen = []
count_consec_error = []
count_consec_error_closer = []
count_consec_error_further = []
count_consec_error_cross = []
count_mob = []
count_consec_mobred = []

hash_diffoptlen = [] # non-surrender
hash_diffoptlen_surrender = []
hash_consec_error = []
hash_consec_error_surrender = []
hash_consec_error_closer = []
hash_consec_error_closer_surrender = []
hash_consec_error_further = []
hash_consec_error_further_surrender = []
hash_consec_error_cross = []
hash_consec_error_cross_surrender = []
hash_mob = []
hash_mob_surrender = []
hash_consec_mobred = []
hash_consec_mobred_surrender = []

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
		min_diffoptlen = int(row['min_diffoptlen'])
		max_diffoptlen = int(row['max_diffoptlen'])
		range_diffoptlen = int(row['range_diffoptlen'])
		max_consec_error = int(row['max_consec_error'])
		max_consec_error_closer = int(row['max_consec_error_closer'])
		max_consec_error_further = int(row['max_consec_error_further'])
		max_consec_error_cross = int(row['max_consec_error_cross'])
		max_mob = int(row['max_mobility'])
		max_consec_mobred = int(row['max_consec_mobility_reduced'])
		surrender_diffoptlen = [0] * (range_diffoptlen + 1)
		count_diffoptlen = [0] * (range_diffoptlen + 1)
		surrender_consec_error = [0] * (max_consec_error + 1)
		count_consec_error = [0] * (max_consec_error + 1)
		surrender_consec_error_closer = [0] * (max_consec_error_closer + 1)
		count_consec_error_closer = [0] * (max_consec_error_closer + 1)
		surrender_consec_error_further = [0] * (max_consec_error_further + 1)
		count_consec_error_further = [0] * (max_consec_error_further + 1)
		surrender_consec_error_cross = [0] * (max_consec_error_cross + 1)
		count_consec_error_cross = [0] * (max_consec_error_cross + 1)
		surrender_mob = [0] * (max_mob + 1)
		count_mob = [0] * (max_mob + 1)
		surrender_consec_mobred = [0] * (max_consec_mobred + 1)
		count_consec_mobred = [0] * (max_consec_mobred + 1)
	cur_ins = row['instance']
	if ins_optlen[all_instances.index(cur_ins)] != optlen_consider: # only consider level-16 puzzles
		continue 
	diffoptlen = row['diffoptlen']
	surrender = row['surrender']
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
	if surrender == 1:
		surrender_diffoptlen[abs(min_diffoptlen) + diffoptlen] += 1
		surrender_consec_error[consec_error] += 1
		surrender_consec_error_closer[consec_error_closer] += 1
		surrender_consec_error_further[consec_error_further] += 1
		surrender_consec_error_cross[consec_error_cross] += 1
		surrender_mob[mobility] += 1
		surrender_consec_mobred[consec_mobred] += 1
		hash_diffoptlen_surrender.append(int(diffoptlen))
		hash_consec_error_surrender.append(int(consec_error))
		hash_consec_error_closer_surrender.append(int(consec_error_closer))
		hash_consec_error_further_surrender.append(int(consec_error_further))
		hash_consec_error_cross_surrender.append(int(consec_error_cross))
		hash_mob_surrender.append(int(mobility))
		hash_consec_mobred_surrender.append(int(consec_mobred))
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


# Consecutive Error Cross: prob surrender and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
# ax[0].hist(hash_consec_error_cross+hash_consec_error_cross_surrender, bins=np.arange(len(count_consec_error_cross))-0.5,\
# 			density=False, align='mid', label='All (left axis)', \
# 			color='olive', edgecolor='olive', alpha=0.5, width=1)
# ax[0].axvline(np.median(hash_consec_error_cross+hash_consec_error_cross_surrender), color='olive', linestyle='dashed', linewidth=1)
ax[0].hist(hash_consec_error_cross, bins=np.arange(len(count_consec_error_cross))-0.5,\
			density=False, align='mid', label='Non-surrender (left axis)', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error_cross), color='gray', linestyle='dashed', linewidth=1)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_cross_surrender, bins=np.arange(len(count_consec_error_cross))-0.5,\
			density=False, align='mid', label='surrender (right axis)', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_cross_surrender), \
		color='blue', linestyle='dashed', linewidth=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.83))
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax[0].set_ylabel('Count', fontsize=14)

cmap = mp.cm.get_cmap('Blues')
normalize = mp.colors.Normalize(vmin=min(count_consec_error_cross), vmax=max(count_consec_error_cross))
colors = [cmap(normalize(value)) for value in count_consec_error_cross]
count_consec_error_cross = np.array(count_consec_error_cross, dtype=np.float32)
surrender_consec_error_cross = np.array(surrender_consec_error_cross, dtype=np.float32)
CIup = (surrender_consec_error_cross+z**2/2.0)/(count_consec_error_cross+z**2)+ (z/(count_consec_error_cross+z**2))*np.sqrt(surrender_consec_error_cross*(count_consec_error_cross-surrender_consec_error_cross)/count_consec_error_cross+z**2/4.0)
CIlow = (surrender_consec_error_cross+z**2/2.0)/(count_consec_error_cross+z**2) - (z/(count_consec_error_cross+z**2))*np.sqrt(surrender_consec_error_cross*(count_consec_error_cross-surrender_consec_error_cross)/count_consec_error_cross+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error_cross)), \
		height=surrender_consec_error_cross/count_consec_error_cross, \
		width=1, alpha=0.65, color=colors, \
		label='Probability surrender', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error_cross)), surrender_consec_error_cross/count_consec_error_cross, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(surrender_consec_error_cross/count_consec_error_cross)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Surrender', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Error Cross', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Surrender; Histogram of Consecutive Error Cross', \
	fontsize=20, weight='bold')
plt.savefig(out_file27)
plt.close()

# hypothesis testing
print('\nSurrender: Consecutive Error Cross')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error_cross, hash_consec_error_cross_surrender,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error_cross, hash_consec_error_cross_surrender,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error_cross))
print('Median Restart: ', np.median(hash_consec_error_cross_surrender))
print('Mean Non-restart: ', np.mean(hash_consec_error_cross))
print('Mean Restart: ', np.mean(hash_consec_error_cross_surrender))








# Consecutive Error Closer: prob surrender and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
# ax[0].hist(hash_consec_error_closer+hash_consec_error_closer_surrender, bins=np.arange(len(count_consec_error_closer))-0.5,\
# 			density=False, align='mid', label='All (left axis)', \
# 			color='olive', edgecolor='olive', alpha=0.5, width=1)
# ax[0].axvline(np.median(hash_consec_error_closer+hash_consec_error_closer_surrender), color='olive', linestyle='dashed', linewidth=1)
ax[0].hist(hash_consec_error_closer, bins=np.arange(len(count_consec_error_closer))-0.5,\
			density=False, align='mid', label='Non-surrender (left axis)', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error_closer), color='gray', linestyle='dashed', linewidth=1)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_closer_surrender, bins=np.arange(len(count_consec_error_closer))-0.5,\
			density=False, align='mid', label='Surrender (right axis)', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_closer_surrender), color='blue', linestyle='dashed', linewidth=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.83))
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax[0].set_ylabel('Count', fontsize=14)

cmap = mp.cm.get_cmap('Blues')
normalize = mp.colors.Normalize(vmin=min(count_consec_error_closer), vmax=max(count_consec_error_closer))
colors = [cmap(normalize(value)) for value in count_consec_error_closer]
count_consec_error_closer = np.array(count_consec_error_closer, dtype=np.float32)
surrender_consec_error_closer = np.array(surrender_consec_error_closer, dtype=np.float32)
CIup = (surrender_consec_error_closer+z**2/2.0)/(count_consec_error_closer+z**2)+ (z/(count_consec_error_closer+z**2))*np.sqrt(surrender_consec_error_closer*(count_consec_error_closer-surrender_consec_error_closer)/count_consec_error_closer+z**2/4.0)
CIlow = (surrender_consec_error_closer+z**2/2.0)/(count_consec_error_closer+z**2) - (z/(count_consec_error_closer+z**2))*np.sqrt(surrender_consec_error_closer*(count_consec_error_closer-surrender_consec_error_closer)/count_consec_error_closer+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error_closer)), \
		height=surrender_consec_error_closer/count_consec_error_closer, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Surrender', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error_closer)), surrender_consec_error_closer/count_consec_error_closer, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(surrender_consec_error_closer/count_consec_error_closer)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Surrender', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Error Closer', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Surrender; Histogram of Consecutive Error Closer', \
	fontsize=20, weight='bold')
plt.savefig(out_file26)
plt.close()

# hypothesis testing
print('\nSurrender: Consecutive Error Closer')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error_closer, hash_consec_error_closer_surrender,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error_closer, hash_consec_error_closer_surrender,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error_closer))
print('Median Restart: ', np.median(hash_consec_error_closer_surrender))
print('Mean Non-restart: ', np.mean(hash_consec_error_closer))
print('Mean Restart: ', np.mean(hash_consec_error_closer_surrender))








# Consecutive Error Further: prob surrender and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
# ax[0].hist(hash_consec_error_further+hash_consec_error_further_surrender, bins=np.arange(len(count_consec_error_further))-0.5,\
# 			density=False, align='mid', label='All (left axis)', \
# 			color='olive', edgecolor='olive', alpha=0.5, width=1)
# ax[0].axvline(np.median(hash_consec_error_further+hash_consec_error_further_surrender), color='olive', linestyle='dashed', linewidth=1)
ax[0].hist(hash_consec_error_further, bins=np.arange(len(count_consec_error_further))-0.5,\
			density=False, align='mid', label='Non-surrender (left axis)', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error_further), color='gray', linestyle='dashed', linewidth=1)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_further_surrender, bins=np.arange(len(count_consec_error_further))-0.5,\
			density=False, align='mid', label='Surrender (right axis)', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_further_surrender), color='blue', linestyle='dashed', linewidth=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.83))
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax[0].set_ylabel('Count', fontsize=14)

cmap = mp.cm.get_cmap('Blues')
normalize = mp.colors.Normalize(vmin=min(count_consec_error_further), vmax=max(count_consec_error_further))
colors = [cmap(normalize(value)) for value in count_consec_error_further]
count_consec_error_further = np.array(count_consec_error_further, dtype=np.float32)
surrender_consec_error_further = np.array(surrender_consec_error_further, dtype=np.float32)
CIup = (surrender_consec_error_further+z**2/2.0)/(count_consec_error_further+z**2)+ (z/(count_consec_error_further+z**2))*np.sqrt(surrender_consec_error_further*(count_consec_error_further-surrender_consec_error_further)/count_consec_error_further+z**2/4.0)
CIlow = (surrender_consec_error_further+z**2/2.0)/(count_consec_error_further+z**2) - (z/(count_consec_error_further+z**2))*np.sqrt(surrender_consec_error_further*(count_consec_error_further-surrender_consec_error_further)/count_consec_error_further+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error_further)), \
		height=surrender_consec_error_further/count_consec_error_further, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Surrender', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error_further)), surrender_consec_error_further/count_consec_error_further, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(surrender_consec_error_further/count_consec_error_further)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Surrender', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Error Further', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Surrender; Histogram of Consecutive Error Further', \
	fontsize=20, weight='bold')
plt.savefig(out_file25)
plt.close()

# hypothesis testing
print('\nSurrender: Consecutive Error Further')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error_further, hash_consec_error_further_surrender,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error_further, hash_consec_error_further_surrender,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error_further))
print('Median Restart: ', np.median(hash_consec_error_further_surrender))
print('Mean Non-restart: ', np.mean(hash_consec_error_further))
print('Mean Restart: ', np.mean(hash_consec_error_further_surrender))










# Consecutive Error: prob surrender and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_consec_error, bins=np.arange(len(count_consec_error))-0.5,\
			density=True, align='mid', label='Non-surrender', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_error), \
			color='gray', linestyle='dashed', linewidth=2.5)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_error_surrender, bins=np.arange(len(count_consec_error))-0.5,\
			density=True, align='mid', label='Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_error_surrender), \
		color='blue', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9), prop={'size': 16})
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.81), prop={'size': 16})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax12.locator_params(nbins=5, axis='y')
ax12.tick_params(axis='both', which='major', labelsize=16, colors='blue')
ax[0].set_ylabel('Count', fontsize=18)
print('total consec_error sample size: ', len(hash_consec_error))
print('total consec_error_restart sample size: ', len(hash_consec_error_surrender))

cmap = mp.cm.get_cmap('Blues')
normalize = mp.colors.Normalize(vmin=min(count_consec_error), vmax=max(count_consec_error))
colors = [cmap(normalize(value)) for value in count_consec_error]
count_consec_error = np.array(count_consec_error, dtype=np.float32)
surrender_consec_error = np.array(surrender_consec_error, dtype=np.float32)
CIup = (surrender_consec_error+z**2/2.0)/(count_consec_error+z**2)+ (z/(count_consec_error+z**2))*np.sqrt(surrender_consec_error*(count_consec_error-surrender_consec_error)/count_consec_error+z**2/4.0)
CIlow = (surrender_consec_error+z**2/2.0)/(count_consec_error+z**2) - (z/(count_consec_error+z**2))*np.sqrt(surrender_consec_error*(count_consec_error-surrender_consec_error)/count_consec_error+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_error)), \
		height=surrender_consec_error/count_consec_error, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Surrender', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_error)), surrender_consec_error/count_consec_error, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(surrender_consec_error/count_consec_error)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Surrender', fontsize=14)


fig.text(0.5, 0.029, \
	'Consecutive Error', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Surrender; Histogram of Consecutive Error', \
	fontsize=20, weight='bold')
plt.savefig(out_file24)
plt.close()

# hypothesis testing
print('\nSurrender: Consecutive Error')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_error, hash_consec_error_surrender,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_error, hash_consec_error_surrender,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_error))
print('Median Restart: ', np.median(hash_consec_error_surrender))
print('Mean Non-restart: ', np.mean(hash_consec_error))
print('Mean Restart: ', np.mean(hash_consec_error_surrender))










# Consecutive Mobility Reduced: prob surrender and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
# ax[0].hist(hash_consec_mobred+hash_consec_mobred_surrender, bins=np.arange(len(count_consec_mobred))-0.5,\
# 			density=False, align='mid', label='All (left axis)', \
# 			color='olive', edgecolor='olive', alpha=0.5, width=1)
# ax[0].axvline(np.median(hash_consec_mobred+hash_consec_mobred_surrender), color='olive', linestyle='dashed', linewidth=1)
ax[0].hist(hash_consec_mobred, bins=np.arange(len(count_consec_mobred))-0.5,\
			density=False, align='mid', label='Non-surrender (left axis)', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_consec_mobred), color='gray', linestyle='dashed', linewidth=1)
ax12 = ax[0].twinx()
ax12.hist(hash_consec_mobred_surrender, bins=np.arange(len(count_consec_mobred))-0.5,\
			density=False, align='mid', label='Surrender (right axis)', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax12.axvline(np.median(hash_consec_mobred_surrender), color='blue', linestyle='dashed', linewidth=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.83))
ax[0].tick_params(axis='both', which='major', labelsize=10)
ax12.tick_params(axis='both', which='major', labelsize=10)
ax[0].set_ylabel('Count', fontsize=14)

cmap = mp.cm.get_cmap('Blues')
normalize = mp.colors.Normalize(vmin=min(count_consec_mobred), vmax=max(count_consec_mobred))
colors = [cmap(normalize(value)) for value in count_consec_mobred]
count_consec_mobred = np.array(count_consec_mobred, dtype=np.float32)
surrender_consec_mobred = np.array(surrender_consec_mobred, dtype=np.float32)
CIup = (surrender_consec_mobred+z**2/2.0)/(count_consec_mobred+z**2)+ (z/(count_consec_mobred+z**2))*np.sqrt(surrender_consec_mobred*(count_consec_mobred-surrender_consec_mobred)/count_consec_mobred+z**2/4.0)
CIlow = (surrender_consec_mobred+z**2/2.0)/(count_consec_mobred+z**2) - (z/(count_consec_mobred+z**2))*np.sqrt(surrender_consec_mobred*(count_consec_mobred-surrender_consec_mobred)/count_consec_mobred+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_consec_mobred)), \
		height=surrender_consec_mobred/count_consec_mobred, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Surrender', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_consec_mobred)), surrender_consec_mobred/count_consec_mobred, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(surrender_consec_mobred/count_consec_mobred)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Surrender', fontsize=14)

fig.text(0.5, 0.029, \
	'Consecutive Mobility Reduced', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Surrender; Histogram of Consecutive Mobility Reduced', \
	fontsize=20, weight='bold')
plt.savefig(out_file23)
plt.close()

# hypothesis testing
print('\nSurrender: Consecutive Mobility Reduced')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_consec_mobred, hash_consec_mobred_surrender,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_consec_mobred, hash_consec_mobred_surrender,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_consec_mobred))
print('Median Restart: ', np.median(hash_consec_mobred_surrender))
print('Mean Non-restart: ', np.mean(hash_consec_mobred))
print('Mean Restart: ', np.mean(hash_consec_mobred_surrender))







# Mobility: prob surrender and histogram
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_mob, bins=np.arange(len(count_mob))-0.5,\
			density=True, align='mid', label='Non-surrender', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_mob), \
			color='gray', linestyle='dashed', linewidth=2.5)
ax12 = ax[0].twinx()
ax12.hist(hash_mob_surrender, bins=np.arange(len(count_mob))-0.5,\
			density=True, align='mid', label='Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax12.axvline(np.median(hash_mob_surrender), \
		color='blue', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9), prop={'size': 16})
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.81), prop={'size': 16})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax12.locator_params(nbins=5, axis='y')
ax12.tick_params(axis='both', which='major', labelsize=16, colors='blue')
ax[0].set_ylabel('Count', fontsize=18)
print('total mob sample size: ', len(hash_mob))
print('total mob_restart sample size: ', len(hash_mob_surrender))

cmap = mp.cm.get_cmap('Blues')
normalize = mp.colors.Normalize(vmin=min(count_mob), vmax=max(count_mob))
colors = [cmap(normalize(value)) for value in count_mob]
count_mob = np.array(count_mob, dtype=np.float32)
surrender_mob = np.array(surrender_mob, dtype=np.float32)
CIup = (surrender_mob+z**2/2.0)/(count_mob+z**2)+ (z/(count_mob+z**2))*np.sqrt(surrender_mob*(count_mob-surrender_mob)/count_mob+z**2/4.0)
CIlow = (surrender_mob+z**2/2.0)/(count_mob+z**2) - (z/(count_mob+z**2))*np.sqrt(surrender_mob*(count_mob-surrender_mob)/count_mob+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_mob)), \
		height=surrender_mob/count_mob, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Surrender', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_mob)), surrender_mob/count_mob, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(surrender_mob/count_mob)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Surrender', fontsize=14)

fig.text(0.5, 0.029, \
	'Mobility', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Surrender; Histogram of Mobility', \
	fontsize=20, weight='bold')
plt.savefig(out_file22)
plt.close()

# hypothesis testing
print('\nSurrender: Mobility')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_mob, hash_mob_surrender,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_mob, hash_mob_surrender,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_mob))
print('Median Restart: ', np.median(hash_mob_surrender))
print('Mean Non-restart: ', np.mean(hash_mob))
print('Mean Restart: ', np.mean(hash_mob_surrender))







# diffoptlen: prob surrender and histogra
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
ax[0].hist(hash_diffoptlen, bins=np.arange(start=min_diffoptlen,stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Non-surrender', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_diffoptlen), \
			color='gray', linestyle='dashed', linewidth=2.5)
ax12 = ax[0].twinx()
ax12.hist(hash_diffoptlen_surrender, bins=np.arange(start=min_diffoptlen,stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax12.axvline(np.median(hash_diffoptlen_surrender), \
		color='blue', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9), prop={'size': 16})
ax12.legend(loc=2, bbox_to_anchor=(0.55,0.81), prop={'size': 16})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax12.locator_params(nbins=5, axis='y')
ax12.tick_params(axis='both', which='major', labelsize=16, colors='blue')
ax[0].set_ylabel('Count', fontsize=18)
print('total diffoptlen sample size: ', len(hash_diffoptlen))
print('total diffoptlen_restart sample size: ', len(hash_diffoptlen_surrender))

cmap = mp.cm.get_cmap('Blues')
normalize = mp.colors.Normalize(vmin=min(count_diffoptlen), vmax=max(count_diffoptlen))
colors = [cmap(normalize(value)) for value in count_diffoptlen]
count_diffoptlen = np.array(count_diffoptlen, dtype=np.float32)
surrender_diffoptlen = np.array(surrender_diffoptlen, dtype=np.float32)
CIup = (surrender_diffoptlen+z**2/2.0)/(count_diffoptlen+z**2)+ (z/(count_diffoptlen+z**2))*np.sqrt(surrender_diffoptlen*(count_diffoptlen-surrender_diffoptlen)/count_diffoptlen+z**2/4.0)
CIlow = (surrender_diffoptlen+z**2/2.0)/(count_diffoptlen+z**2) - (z/(count_diffoptlen+z**2))*np.sqrt(surrender_diffoptlen*(count_diffoptlen-surrender_diffoptlen)/count_diffoptlen+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_diffoptlen)), \
		height=surrender_diffoptlen/count_diffoptlen, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Surrender', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_diffoptlen)), surrender_diffoptlen/count_diffoptlen, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=4, color=color)
ax[1].set_ylim(top=np.nanmax(surrender_diffoptlen/count_diffoptlen)+0.01)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('Probability Surrender', fontsize=14)

fig.text(0.5, 0.029, \
	'Difference of Current Length and Initial Optlen', \
	ha='center', fontsize=15)
plt.suptitle('Probability of Surrender; Histogram of Length Difference', \
	fontsize=20, weight='bold')
plt.savefig(out_file21)
plt.close()

# hypothesis testing
print('\nSurrender: Length Difference')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_diffoptlen, hash_diffoptlen_surrender,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_diffoptlen, hash_diffoptlen_surrender,\
						equal_var=False, nan_policy='omit'))
print('Median Non-restart: ', np.median(hash_diffoptlen))
print('Median Restart: ', np.median(hash_diffoptlen_surrender))
print('Mean Non-restart: ', np.mean(hash_diffoptlen))
print('Mean Restart: ', np.mean(hash_diffoptlen_surrender))















sys.exit()
##################################### OLD PLOT ##################################

out_file10 = '/Users/chloe/Desktop/prob_surrender_error.png'
out_file11 = '/Users/chloe/Desktop/density_surrender_error.png'
out_file12 = '/Users/chloe/Desktop/prob_surrender_density_diffoptlen.png'

out_file13 = '/Users/chloe/Desktop/prob_surrender_mob.png'
out_file14 = '/Users/chloe/Desktop/density_surrender_mob.png'

out_file15 = '/Users/chloe/Desktop/count_surrender_error.png'
out_file16 = '/Users/chloe/Desktop/count_surrender_mob.png'
# density of mobility and consec_mobred
fig, ax = plt.subplots(2, 2, figsize=(15, 16))
ax[0, 0].hist(hash_mob, bins=np.arange(len(surrender_mob))-0.5,\
			density=False, align='mid', label='Mobility', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_xlabel('Mobility', fontsize=18)
ax[0, 1].hist(hash_mob_surrender, bins=np.arange(len(surrender_mob))-0.5,\
			density=False, align='mid', label='Mobility surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_xlabel('Mobility Surrender', fontsize=18)

ax[1, 0].hist(hash_consec_mobred, bins=np.arange(len(surrender_consec_mobred))-0.5,\
			density=False, align='mid', label='Consecutive Mobility Reduced', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_xlabel('Consecutive Mobility Reduced', fontsize=18)
ax[1, 1].hist(hash_consec_mobred_surrender, bins=np.arange(len(surrender_consec_mobred))-0.5,\
			density=False, align='mid', label='Consecutive Mobility Reduced surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_xlabel('Consecutive Mobility Reduced Surrender', fontsize=18)

fig.text(0.07, 0.5, 'Count', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Histogram of Mobility and Consecutive Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file16)
plt.close()

sys.exit()



# count histogram of consec errors with surrender
fig, ax = plt.subplots(4, 2, figsize=(15, 8))
ax[0, 0].hist(hash_consec_error, bins=np.arange(len(surrender_consec_error))-0.5,\
			density=False, align='mid', label='Overall', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)
ax[0, 1].hist(hash_consec_error_surrender, bins=np.arange(len(surrender_consec_error))-0.5,\
			density=False, align='mid', label='Overall surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_title('Overall Surrender', fontsize=15)

ax[1, 0].hist(hash_consec_error_cross, bins=np.arange(len(surrender_consec_error_cross))-0.5,\
			density=False, align='mid', label='Cross', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)
ax[1, 1].hist(hash_consec_error_cross_surrender, bins=np.arange(len(surrender_consec_error_cross))-0.5,\
			density=False, align='mid', label='Cross surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Cross Surrender', fontsize=15)

ax[2, 0].hist(hash_consec_error_further, bins=np.arange(len(surrender_consec_error_further))-0.5,\
			density=False, align='mid', label='Further', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[2, 0].tick_params(axis='both', which='major', labelsize=13)
ax[2, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))
ax[2, 0].set_title('Further', fontsize=15)
ax[2, 1].hist(hash_consec_error_further_surrender, bins=np.arange(len(surrender_consec_error_further))-0.5,\
			density=False, align='mid', label='Further surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[2, 1].tick_params(axis='both', which='major', labelsize=13)
ax[2, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
ax[2, 1].set_title('Further Surrender', fontsize=15)

ax[3, 0].hist(hash_consec_error_closer, bins=np.arange(len(surrender_consec_error_closer))-0.5,\
			density=False, align='mid', label='Closer', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[3, 0].tick_params(axis='both', which='major', labelsize=13)
ax[3, 0].set_title('Closer', fontsize=15)
ax[3, 1].hist(hash_consec_error_closer_surrender, bins=np.arange(len(surrender_consec_error_closer))-0.5,\
			density=False, align='mid', label='Closer surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[3, 1].tick_params(axis='both', which='major', labelsize=13)
ax[3, 1].set_title('Closer Surrender', fontsize=15)

fig.text(0.5, 0.06, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.045, 0.5, 'Count', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Histogram of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file15)
plt.close()

# sys.exit()



# prob surrender of mobility and consec_mobred
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(len(surrender_mob)), \
		height=np.array(surrender_mob, dtype=np.float32) / np.array(count_mob, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Surrender', align='center')
# ax[0].xaxis.set_major_locator(ticker.MultipleLocator(7))
ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_xlabel('Mobility', fontsize=18)

ax[1].bar(x=np.arange(len(surrender_consec_mobred)), \
		height=np.array(surrender_consec_mobred, dtype=np.float32) / np.array(count_consec_mobred, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Surrender', align='center')
# ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_xlabel('Consecutive Mobility Reduced', fontsize=18)

fig.text(0.06, 0.5, 'Probability Surrender', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Probability Surrender as a Function of Mobility and Consecutive Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file13)
plt.close()

# density of mobility and consec_mobred with surrender density
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].hist(hash_mob, bins=np.arange(len(surrender_mob))-0.5,\
			density=True, align='mid', label='Mobility', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].hist(hash_mob_surrender, bins=np.arange(len(surrender_mob))-0.5,\
			density=True, align='mid', label='Mobility Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_xlabel('Mobility', fontsize=18)

ax[1].hist(hash_consec_mobred, bins=np.arange(len(surrender_consec_mobred))-0.5,\
			density=True, align='mid', label='Consecutive Mobility Reduced', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_consec_mobred_surrender, bins=np.arange(len(surrender_consec_mobred))-0.5,\
			density=True, align='mid', label='Consecutive Mobility Reduced Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[1].legend(loc=2, bbox_to_anchor=(0.35,0.9))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_xlabel('Consecutive Mobility Reduced', fontsize=18)

fig.text(0.06, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Density of Mobility and Consecutive Mobility Reduced', fontsize=20, weight='bold')
plt.savefig(out_file14)
plt.close()


# diffoptlen: prob surrender and density histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(x=np.arange(start=min_diffoptlen, stop=max_diffoptlen+1), \
		height=np.array(surrender_diffoptlen, dtype=np.float32) / np.array(count_diffoptlen, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Surrender', align='center')
ax[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].set_ylabel('Probability Surrender', fontsize=18)

ax[1].hist(hash_diffoptlen, bins=np.arange(start=min_diffoptlen, stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Diffoptlen', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1].hist(hash_diffoptlen_surrender, bins=np.arange(start=min_diffoptlen, stop=max_diffoptlen+1)-0.5,\
			density=True, align='mid', label='Diffoptlen Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_ylabel('Density', fontsize=18)

fig.text(0.5, 0.03, 'Diffoptlen', ha='center', fontsize=18)
plt.suptitle('Probability of Surrender by and Density of Length Difference', fontsize=20, weight='bold')
plt.savefig(out_file12)
plt.close()

sys.exit()


# density of consec errors with surrender
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].hist(hash_consec_error, bins=np.arange(len(surrender_consec_error))-0.5,\
			density=True, align='mid', label='Overall', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 0].hist(hash_consec_error_surrender, bins=np.arange(len(surrender_consec_error))-0.5,\
			density=True, align='mid', label='Overall Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[0, 0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)

ax[1, 0].hist(hash_consec_error_cross, bins=np.arange(len(surrender_consec_error_cross))-0.5,\
			density=True, align='mid', label='Cross', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 0].hist(hash_consec_error_cross_surrender, bins=np.arange(len(surrender_consec_error_cross))-0.5,\
			density=True, align='mid', label='Cross Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[1, 0].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)

ax[0, 1].hist(hash_consec_error_further, bins=np.arange(len(surrender_consec_error_further))-0.5,\
			density=True, align='mid', label='Further', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0, 1].hist(hash_consec_error_further_surrender, bins=np.arange(len(surrender_consec_error_further))-0.5,\
			density=True, align='mid', label='Further Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[0, 1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_title('Further', fontsize=15)

ax[1, 1].hist(hash_consec_error_closer, bins=np.arange(len(surrender_consec_error_closer))-0.5,\
			density=True, align='mid', label='Closer', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[1, 1].hist(hash_consec_error_closer_surrender, bins=np.arange(len(surrender_consec_error_closer))-0.5,\
			density=True, align='mid', label='Closer Surrender', \
			color='blue', edgecolor='blue', alpha=0.3, width=1)
ax[1, 1].legend(loc=2, bbox_to_anchor=(0.55,0.9))
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Closer', fontsize=15)

fig.text(0.5, 0.04, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.07, 0.5, 'Density', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Density of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file11)
plt.close()



# prob surrender of consec errors
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0, 0].bar(x=np.arange(len(surrender_consec_error)), \
		height=np.array(surrender_consec_error, dtype=np.float32) / np.array(count_consec_error, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Surrender', align='center')
ax[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(8))
ax[0, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
ax[0, 0].set_title('Overall', fontsize=15)

ax[1, 0].bar(x=np.arange(len(surrender_consec_error_cross)), \
		height=np.array(surrender_consec_error_cross, dtype=np.float32) / np.array(count_consec_error_cross, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Surrender', align='center')
ax[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(8))
ax[1, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
ax[1, 0].set_title('Cross', fontsize=15)

ax[0, 1].bar(x=np.arange(len(surrender_consec_error_further)), \
		height=np.array(surrender_consec_error_further, dtype=np.float32) / np.array(count_consec_error_further, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Surrender', align='center')
ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(6))
ax[0, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
ax[0, 1].set_title('Further', fontsize=15)

ax[1, 1].bar(x=np.arange(len(surrender_consec_error_closer)), \
		height=np.array(surrender_consec_error_closer, dtype=np.float32) / np.array(count_consec_error_closer, dtype=np.float32), \
		width=1, color='gray', edgecolor='black', alpha=0.7,
		label='Probability Surrender', align='center')
ax[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(7))
ax[1, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
ax[1, 1].set_title('Closer', fontsize=15)

fig.text(0.5, 0.04, 'Consecutive Error Count', ha='center', fontsize=18)
fig.text(0.07, 0.5, 'Probability Surrender', va='center', rotation='vertical', fontsize=18)
plt.suptitle('Probability Surrender as a Function of Consecutive Errors', fontsize=20, weight='bold')
plt.savefig(out_file10)
plt.close()



