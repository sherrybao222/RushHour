# record which subject did which puzzles
import json, math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG, solution
from scipy import stats

path_file = '/Users/chloe/Documents/RushHour/exp_data/paths.json'
data_out = '/Users/chloe/Documents/RushHour/exp_data/dynamic_MAG/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
fig_out = '/Users/chloe/Documents/RushHour/state_figures/sub_puz.png'
data_out = '/Users/chloe/Documents/RushHour/state_model/in_data2/sub_puz.npy'
data = []
sub_dict = {}
sub_list = [] # num_sub x num_puzzle

with open(path_file) as f: # read path data
	for line in f:
		data.append(json.loads(line))
for i in range(0, len(data)): # each line of file
	line = data[i]
	instance = line['instance']
	subject = line['subject']
	complete = line['complete']
	human_len = line['human_length']
	if subject not in sub_dict:
		sub_dict[subject] = [np.NAN]*len(all_instances)
	# fill in tried instance with max int
	sub_dict[subject][all_instances.index(instance)] = 100
	if complete == 'False':
		continue
	# fill in human len if completed
	sub_dict[subject][all_instances.index(instance)] = int(human_len)
for k in sub_dict: # convert dict to list
	sub_list.append(sub_dict[k])


# plot matrix
plt.matshow(np.array(sub_list), vmin=0, vmax=40, cmap='binary')
plt.ylabel('Subjects', fontsize=10)
plt.xlabel('Puzzles', fontsize=10)
plt.suptitle('Subjects vs Puzzles', fontsize=12)
plt.title('coded by subject length', fontsize=9, x=0.55	, y=1.1)
# plt.show()
plt.colorbar()
plt.savefig(fig_out)
# count number of subjects for each puzzle
# print(np.sum(sub_list, axis=0)) # [1, 70]
print(np.array(sub_list).shape) # [86, 70] = [#sub, #puz]
# save output
np.save(data_out, np.array(sub_list))


'''
number of subjects for each puzzle
[45 41 43 42 41 34 48 34 34 40 39 39 43 42 37 37 43 46
 43 47 46 44 41 43 37 43 39 39 46 39 44 47 39 42 39 43 
 40 41 50 41 38 42 43 41 44 43 38 37 39 40 40 37 41 
 42 42 36 42 40 41 41 39 38 38 40 40 39 41 38 41 37]
'''
