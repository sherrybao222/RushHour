#record which subject did which puzzles
import json, math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG, solution
from scipy import stats


path_file = '/Users/chloe/Documents/RushHour/exp_data/paths_filtered.json'
data_out = '/Users/chloe/Documents/RushHour/exp_data/dynamic_MAG/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
fig_out = '/Users/chloe/Desktop/sub_puz.png'
data_out = '/Users/chloe/Documents/RushHour/sub_puz.npy'

data = []
valid_subjects = np.load('/Users/chloe/Documents/RushHour/exp_data/valid_sub.npy')
sub_list = np.zeros((len(valid_subjects), len(all_instances))) # initialize with -1
print(sub_list.shape)

# read path data
with open(path_file) as f: 
	for line in f:
		data.append(json.loads(line))

# process each line of data
for i in range(0, len(data)): 
	line = data[i]
	instance = line['instance']
	subject = line['subject']+':'+line['assignment']
	complete = line['complete']
	human_len = line['human_length']
	if complete == 'False': # surrender/restart
		sub_list[np.where(valid_subjects==subject)[0][0]][all_instances.index(instance)]=np.nan
	else: # success
		sub_list[np.where(valid_subjects==subject)[0][0]][all_instances.index(instance)]=int(human_len)


# plot matrix
masked_array = np.ma.array(sub_list, mask=np.isnan(sub_list))
current_cmap = matplotlib.cm.Blues
current_cmap.set_bad(color='red',alpha=0.5)
plt.matshow(masked_array, interpolation='nearest', vmin=0, vmax=35, cmap=current_cmap)
plt.ylabel('Subjects', fontsize=10)
plt.xlabel('Puzzles', fontsize=10)
plt.suptitle('Subjects vs Puzzles', fontsize=12)
plt.title('Color Coded by Subject Solution Length (Pink: Surrender, White: Not Occured)', fontsize=9, x=0.55	, y=1.1)
plt.colorbar()
plt.savefig(fig_out)
# count number of subjects for each puzzle
print(np.array(sub_list).shape) # [86, 70] = [#sub, #puz]
# save output
np.save(data_out, np.array(sub_list))

# print statistics
bin_matrix = np.array(masked_array, dtype=bool)
print(masked_array)
print(bin_matrix[-2][-1])
print('Each subject on average did '+str(np.mean(np.sum(bin_matrix, axis=1)))+' puzzles.')
print('STD '+str(np.std(np.sum(bin_matrix, axis=1)))+', Median '+str(np.median(np.sum(bin_matrix,axis=1))))
print('Each puzzle has on average '+str(np.mean(np.sum(bin_matrix, axis=0)))+' subjects.')
print('STD '+str(np.std(np.sum(bin_matrix, axis=0)))+', Median '+str(np.median(np.sum(bin_matrix,axis=0))))


'''
number of subjects for each puzzle
[45 41 43 42 41 34 48 34 34 40 39 39 43 42 37 37 43 46
 43 47 46 44 41 43 37 43 39 39 46 39 44 47 39 42 39 43 
 40 41 50 41 38 42 43 41 44 43 38 37 39 40 40 37 41 
 42 42 36 42 40 41 41 39 38 38 40 40 39 41 38 41 37]
'''
