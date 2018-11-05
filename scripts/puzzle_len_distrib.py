# visualize human length distribution for each puzzle (from puzzle path data)
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

datafile = '/Users/chloe/Documents/RushHour/data/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/'
data = []

# iterate through each puzzle
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	ins_path_file = datafile + instance + '_paths.json'
	ins_distrib = [0] * 10 
	#print(ins_distrib)
	opt_len = 0
	# 0[0,7], 1(7,11], 2(11,14], 3(14,16], 4(16,25], 5(25,35], 6(35,45], 7(45-55], 8(55,65], 9(65s, +)
	# load json data
	with open(ins_path_file) as f:
		for line in f:
			cur_data = json.loads(line)
			if cur_data['complete'] == 'False':
				continue
			human_len = int(cur_data['human_length'])
			opt_len = int(cur_data['optimal_length'])
			if 0 <= human_len <= 7:
				ins_distrib[0] += 1
			elif 7 < human_len <= 11:
				ins_distrib[1] += 1
			elif 11 < human_len <= 14:
				ins_distrib[2] += 1
			elif 14 < human_len <= 16:
				ins_distrib[3] += 1
			elif 16 < human_len <= 25:
				ins_distrib[4] += 1
			elif 25 < human_len <= 35:
				ins_distrib[5] += 1
			elif 35 < human_len <= 45:
				ins_distrib[6] += 1
			elif 45 < human_len <= 55:
				ins_distrib[7] += 1
			elif 55 < human_len <= 65:
				ins_distrib[8] += 1
			elif 65 < human_len:
				ins_distrib[9] += 1
	# visualize distribution for current puzzle
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x_axis = ['0-7', '8-11', '12-14', '15-16', '17-25', '26-35', '36-45', '46-55', '56-65', '65+']
	ax.bar(x_axis, ins_distrib, alpha=0.9, color='green')
	ax.set_xlabel('#moves human take')
	ax.set_ylabel('frequency')
	plt.title('Human len frequency puzzle ' + instance + ', opt_len=' + str(opt_len))
	fig_out_dir = out_dir + instance + '/' + instance + '_hum_len_distr.png'
	ax.grid(axis = 'y', alpha = 0.3)
	plt.savefig(fig_out_dir)
	plt.close()







