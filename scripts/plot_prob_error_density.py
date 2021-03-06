# PLOTTING: characterize trialdata each move
# probability of error as a function of MAG size (#nodes)
# density histograms of MAG size
# py27
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
optlen_consider = 7

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
out_file = '/Users/chloe/Desktop/nodes_to_error_6.png'


move_data = pd.read_csv(moves_file)

error_node = []

count_node = []

hash_node = []
hash_node_error = []

max_node = ''


#################################### PROCESS DATA ###############################

for i in range(len(move_data)):
	row = move_data.loc[i, :]
	# first line
	if i == 0: 
		max_node = int(row['max_node'])
		error_node = [0]* (max_node + 1)
		count_node = [0] * (max_node + 1)
	cur_ins = row['instance']
	if ins_optlen[all_instances.index(cur_ins)] != optlen_consider: 
	# only consider designated level
		continue 
	error = row['error_tomake']
	node = row['node_human_static']
	count_node[node] += 1
	if error == 1:
		error_node[node] += 1
		hash_node_error.append(int(node))
	else:
		hash_node.append(int(node))

####################################### PLOTTING ##################################
# error bar preparation (95%CI)
z = st.norm.ppf(1-0.05/2)


# node: prob error and histogram
fig, ax = plt.subplots(2,1, figsize=(30, 40))
ax[0].hist(hash_node, bins=np.arange(len(count_node))-0.5,\
			density=True, align='mid', label='Non-Error', \
			color='gray', edgecolor='black', alpha=0.7, width=1)
ax[0].axvline(np.median(hash_node), \
			color='gray', linestyle='dashed', linewidth=2.5)
# ax12 = ax[0].twinx()
# ax12.hist(hash_node_error, bins=np.arange(len(count_node))-0.5,\
# 			density=True, align='mid', label='Error', \
# 			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
# ax12.axvline(np.median(hash_node_error), \
# 		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].hist(hash_node_error, bins=np.arange(len(count_node))-0.5,\
			density=True, align='mid', label='Error', \
			color='orangered', edgecolor='orangered', alpha=0.3, width=1)
ax[0].axvline(np.median(hash_node_error), \
		color='orangered', linestyle='dashed', linewidth=2.5)
ax[0].legend(loc='upper center', bbox_to_anchor=(0.55,0.9), prop={'size': 60})
# ax12.legend(loc='upper center', bbox_to_anchor=(0.55,0.81), prop={'size': 50})
ax[0].locator_params(nbins=5, axis='y')
ax[0].tick_params(axis='both', which='major', labelsize=80)
# ax12.locator_params(nbins=5, axis='y')
# ax12.tick_params(axis='both', which='major', labelsize=60, colors='orangered')
ax[0].set_ylabel('Proportion of Positions', fontsize=80)
ax[0].set_xlabel('Graph Size', fontsize=80)
ax[0].set_title('Histogram of Graph Size Given Error/Correct', \
				fontsize=30, weight='bold')
print('total node sample size: ', len(hash_node))
print('total node_error sample size: ', len(hash_node_error))


cmap = mp.cm.get_cmap('OrRd')
normalize = mp.colors.Normalize(vmin=min(count_node), vmax=max(count_node))
colors = [cmap(normalize(value)) for value in count_node]
count_node = np.array(count_node, dtype=np.float32)
error_node = np.array(error_node, dtype=np.float32)
CIup = (error_node+z**2/2.0)/(count_node+z**2)+ (z/(count_node+z**2))*np.sqrt(error_node*(count_node-error_node)/count_node+z**2/4.0)
CIlow = (error_node+z**2/2.0)/(count_node+z**2) - (z/(count_node+z**2))*np.sqrt(error_node*(count_node-error_node)/count_node+z**2/4.0)		
ax[1].bar(x=np.arange(len(count_node)), \
		height=error_node/count_node, \
		width=1, alpha=0.65, color=colors, \
		label='Probability Error', align='center') 
ax[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
for pos, y, err, color in zip(np.arange(len(count_node)), error_node/count_node, np.array(zip(CIlow,CIup)), colors):
    err=np.expand_dims(err,axis=1)
    ax[1].errorbar(pos, y, err, capsize=14, color=color, linewidth=8)
ax[1].set_ylim(top=np.nanmax(error_node/count_node)+0.55)
ax[1].tick_params(axis='both', which='major', labelsize=80)
ax[1].set_ylabel('Probability Error', fontsize=80)
ax[1].set_xlabel('Graph Size', fontsize=80)
ax[1].set_title('Probability of Error Given Graph Size', \
				fontsize=20, weight='bold')

# fig.text(0.5, 0.029, \
# 	'Graph Size', \
# 	ha='center', fontsize=40)
# plt.suptitle('Length-'+str(optlen_consider-1)+' Puzzle', \
# 	fontsize=40, weight='bold')
plt.savefig(out_file)
plt.close()

# hypothesis testing
print('\nerror: node')
print('Mann-Whitney U Test: ', \
		st.mannwhitneyu(hash_node, hash_node_error,\
						use_continuity=False))
print('T-test Independent: ', \
		st.ttest_ind(hash_node, hash_node_error,\
						equal_var=False, nan_policy='omit'))
print('Median Non-error: ', np.median(hash_node))
print('Median error: ', np.median(hash_node_error))
print('Mean Non-error: ', np.mean(hash_node))
print('Mean error: ', np.mean(hash_node_error))









sys.exit()