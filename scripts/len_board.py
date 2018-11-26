# puzzle level (initial state): mean human length, optimal length, board info
# visualize bar plots and scatter plots for correlation and p
# save features data files
import json, math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
from scipy import stats

len_file = '/Users/chloe/Documents/RushHour/exp_data/paths.json'
ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
data_out = '/Users/chloe/Documents/RushHour/exp_data/board_info/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
bar_out_dir = '/Users/chloe/Documents/RushHour/puzzle_figures/len_board.png'
scatter_out = '/Users/chloe/Documents/RushHour/puzzle_figures/len_board_scatter.png'
scatter_x = 2
scatter_y = 2
num_features = 1

label_list = ['human_len', 'opt_len', 'board_freedom']
feature_list = ['y_human', 'y_opt', 'y_freedom']
dict_list = [{}] * num_features
y_list = [[]] * num_features
y_human = []
y_opt = []

data = []
humanlen_dict = {} # human_len
mean_dict = {} # mean human_len
optimal_dict = {} # opt_len
all_human_len = [] # all human length for every puzzle
y_human_err = [] # error bar

# preprocess human & optimal len dict
for i in range(len(all_instances)):
	humanlen_dict[all_instances[i] + '_count'] = 0
	humanlen_dict[all_instances[i]+ '_len'] = 0
	all_human_len.append([])
	mean_dict[all_instances[i]] = 0
	optimal_dict[all_instances[i]] = 0
with open(len_file) as f:
	for line in f:
		data.append(json.loads(line))
for i in range(0, len(data)): # iterate through every subject trial
	line = data[i]
	instance = line['instance']
	complete = line['complete']
	human_len = int(line['human_length'])
	opt_len = int(line['optimal_length'])
	if complete == 'False':
		continue
	else:
		humanlen_dict[instance + '_count'] += 1
		humanlen_dict[instance + '_len'] = humanlen_dict[instance + '_len'] + human_len
		ins_index = all_instances.index(instance)
		all_human_len[ins_index].append(human_len)
		optimal_dict[instance] = opt_len
for i in range(len(all_instances)): # calculate mean human len and std
	if humanlen_dict[all_instances[i] + '_count'] == 0:
		mean_dict[all_instances[i]] = 0
		continue
	else:
		mean_dict[all_instances[i]] = humanlen_dict[all_instances[i] + '_len'] / humanlen_dict[all_instances[i] + '_count']
	y_human_err.append(np.std(all_human_len[i]) / math.sqrt(humanlen_dict[all_instances[i]+ '_count']))


# process mag attributes
for i in range(len(all_instances)):
	# construct mag
	cur_ins = all_instances[i]
	ins_dir = ins_file + all_instances[i] + '.json'
	my_car_list, my_red = MAG.json_to_car_list(ins_dir)
	my_board, my_red = MAG.construct_board(my_car_list)
	dict_list[0][cur_ins] = MAG.board_freedom(my_board)


# generate value lists
for i in range(len(all_instances)):
	y_human.append(mean_dict[all_instances[i]])
	y_opt.append(optimal_dict[all_instances[i]])
	for j in range(len(y_list)):
		y_list[j].append(dict_list[j][all_instances[i]])
# save data
np.save(data_out + 'y_human.npy',y_human) # mean human len
np.save(data_out + 'y_opt.npy', y_opt) # opt len
for i in range(2,len(feature_list)):
	np.save(data_out + feature_list[i] + '.npy', y_list[i-2])


# calculate pearson correlation and p-value for human_len and MAG info
corr_list = []
p_list = []
for i in range(2, len(label_list)):
	corr, p = stats.spearmanr(y_human, y_list[i-2])
	corr_list.append(corr)
	p_list.append(p)
	print(('SP-corr human_len & ' + label_list[i] + ': %s, P-value is %s\n') % (str(format(corr, '.2g')), str(format(p, '.2g'))))


# create scatter plot to show correlation and p
fig, axarr = plt.subplots(scatter_x, scatter_y, figsize=(scatter_x*8, scatter_y*8))
count = 0
for i in range(scatter_x):
	for j in range(scatter_y):
		if count >= len(y_list):
			axarr[i,j].axis('off')
			continue
		axarr[i,j].scatter(y_human, y_list[count])
		axarr[i,j].set_ylabel(label_list[count+2], fontsize=20, fontweight='bold')
		t = 'rho=%s, p=%s'%(format(corr_list[0], '.3f'), format(p_list[0], '.2g'))
		axarr[i,j].set_title(t,y=0.85,fontsize=17)
		axarr[i,j].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
		axarr[i,j].tick_params(axis='both', labelsize=17)
		count += 1

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1, rect=None) 
plt.suptitle('human_len vs board', y=0.999, fontsize=22, fontweight='bold')
# plt.show()
plt.savefig(scatter_out)
plt.close()



