# visualize puzzle-level human solving time distribution (from puzzle path data)
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
	ins_distrib = [0] * 11
	#print(ins_distrib)
	opt_len = 0
	# '0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300+'
	# load json data
	with open(ins_path_file) as f:
		for line in f:
			cur_data = json.loads(line)
			if cur_data['complete'] == 'False':
				continue
			time = float(cur_data['rt'])
			time = (time / 1000) # convert to seconds
			if time > 60:
				print(time)
			opt_len = int(cur_data['optimal_length'])
			if 0 <= time <= 30:
				ins_distrib[0] += 1
			elif 30 < time <= 60:
				ins_distrib[1] += 1
			elif 60 < time <= 90:
				ins_distrib[2] += 1
			elif 90 < time <= 120:
				ins_distrib[3] += 1
			elif 120 < time <= 150:
				ins_distrib[4] += 1
			elif 150 < time <= 180:
				ins_distrib[5] += 1
			elif 180 < time <= 210:
				ins_distrib[6] += 1
			elif 210 < time <= 240:
				ins_distrib[7] += 1
			elif 240 < time <= 270:
				ins_distrib[8] += 1
			elif 270 < time <= 300:
				ins_distrib[9] += 1
			elif 300 < time:
				ins_distrib[10] += 1
	# visualize distribution for current puzzle
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x_axis = ['30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '300+']
	ax.bar(x_axis, ins_distrib, alpha=0.9, color='orange')
	ax.set_xlabel('#time human take')
	ax.set_ylabel('frequency')
	plt.title('Human time frequency puzzle ' + instance + ', opt_len=' + str(opt_len))
	fig_out_dir = out_dir + instance + '/' + instance + '_hum_time_distr.png'
	ax.grid(axis = 'y', alpha = 0.3)
	plt.savefig(fig_out_dir)
	plt.close()
