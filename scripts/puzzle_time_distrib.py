# visualize puzzle-level human solving time distribution (from puzzle path data)
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from numpy import median

datafile = '/Users/chloe/Documents/RushHour/data/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/'
data = []

# iterate through each puzzle
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	ins_path_file = datafile + instance + '_paths.json'
	ins_distrib = [0] * 11 # first success time
	ins_surr_distrib = [0] * 11 # surrender trial time
	ins_rest_distrib = [0] * 11 # restart but not complete
	ins_rc1_distrib = [0] * 11 # restart once and complete 
	ins_rcm_distrib = [0] * 11 # restart more than once and complete 
	time_list = [] 
	#print(ins_distrib)
	opt_len = 0
	# '0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270', '270-300', '300+'
	# load json data
	with open(ins_path_file) as f:
		prev_sub = ''
		restart_more_list = []
		for line in f:
			cur_data = json.loads(line)
			time = float(cur_data['rt'])
			time = (time / 1000) # convert to seconds
			time_list.append(time)
			opt_len = int(cur_data['optimal_length'])
			cur_sub = cur_data['subject']
			if cur_data['complete'] == 'False':
				if cur_data['skipped'] == 'True': # surrender
					if 0 <= time <= 30:
						ins_surr_distrib[0] += 1
					elif 30 < time <= 60:
						ins_surr_distrib[1] += 1
					elif 60 < time <= 90:
						ins_surr_distrib[2] += 1
					elif 90 < time <= 120:
						ins_surr_distrib[3] += 1
					elif 120 < time <= 150:
						ins_surr_distrib[4] += 1
					elif 150 < time <= 180:
						ins_surr_distrib[5] += 1
					elif 180 < time <= 210:
						ins_surr_distrib[6] += 1
					elif 210 < time <= 240:
						ins_surr_distrib[7] += 1
					elif 240 < time <= 270:
						ins_surr_distrib[8] += 1
					elif 270 < time <= 300:
						ins_surr_distrib[9] += 1
					elif 300 < time:
						ins_surr_distrib[10] += 1
				else: # restart but not complete
					if 0 <= time <= 30:
						ins_rest_distrib[0] += 1
					elif 30 < time <= 60:
						ins_rest_distrib[1] += 1
					elif 60 < time <= 90:
						ins_rest_distrib[2] += 1
					elif 90 < time <= 120:
						ins_rest_distrib[3] += 1
					elif 120 < time <= 150:
						ins_rest_distrib[4] += 1
					elif 150 < time <= 180:
						ins_rest_distrib[5] += 1
					elif 180 < time <= 210:
						ins_rest_distrib[6] += 1
					elif 210 < time <= 240:
						ins_rest_distrib[7] += 1
					elif 240 < time <= 270:
						ins_rest_distrib[8] += 1
					elif 270 < time <= 300:
						ins_rest_distrib[9] += 1
					elif 300 < time:
						ins_rest_distrib[10] += 1
					if cur_sub == prev_sub and cur_sub not in restart_more_list:
						restart_more_list.append(cur_sub)
			else: # complete
				if cur_sub != prev_sub: # first time success/complete
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
				else: # restart and complete
					if cur_sub in restart_more_list: # restart more than once and complete
						if 0 <= time <= 30:
							ins_rcm_distrib[0] += 1
						elif 30 < time <= 60:
							ins_rcm_distrib[1] += 1
						elif 60 < time <= 90:
							ins_rcm_distrib[2] += 1
						elif 90 < time <= 120:
							ins_rcm_distrib[3] += 1
						elif 120 < time <= 150:
							ins_rcm_distrib[4] += 1
						elif 150 < time <= 180:
							ins_rcm_distrib[5] += 1
						elif 180 < time <= 210:
							ins_rcm_distrib[6] += 1
						elif 210 < time <= 240:
							ins_rcm_distrib[7] += 1
						elif 240 < time <= 270:
							ins_rcm_distrib[8] += 1
						elif 270 < time <= 300:
							ins_rcm_distrib[9] += 1
						elif 300 < time:
							ins_rcm_distrib[10] += 1
					else: # restart once and complete
						if 0 <= time <= 30:
							ins_rc1_distrib[0] += 1
						elif 30 < time <= 60:
							ins_rc1_distrib[1] += 1
						elif 60 < time <= 90:
							ins_rc1_distrib[2] += 1
						elif 90 < time <= 120:
							ins_rc1_distrib[3] += 1
						elif 120 < time <= 150:
							ins_rc1_distrib[4] += 1
						elif 150 < time <= 180:
							ins_rc1_distrib[5] += 1
						elif 180 < time <= 210:
							ins_rc1_distrib[6] += 1
						elif 210 < time <= 240:
							ins_rc1_distrib[7] += 1
						elif 240 < time <= 270:
							ins_rc1_distrib[8] += 1
						elif 270 < time <= 300:
							ins_rc1_distrib[9] += 1
						elif 300 < time:
							ins_rc1_distrib[10] += 1
			prev_sub = cur_sub
	# convert to portion
	ins_distrib = np.array(ins_distrib) / float(len(time_list))
	ins_rest_distrib = np.array(ins_rest_distrib) / float(len(time_list))
	ins_surr_distrib = np.array(ins_surr_distrib) / float(len(time_list))
	ins_rc1_distrib = np.array(ins_rc1_distrib) / float(len(time_list))
	ins_rcm_distrib = np.array(ins_rcm_distrib) / float(len(time_list))
	# print(sum(ins_distrib)+sum(ins_rest_distrib)+sum(ins_surr_distrib)\
			# +sum(ins_rc1_distrib)+sum(ins_rcm_distrib))
	# visualize distribution for current puzzle
	fig = plt.figure(figsize=(5.5,6))
	ax = fig.add_subplot(111)
	x_axis = ['30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '+']
	ax.bar(x_axis, ins_distrib, width=-0.8, align='edge',alpha=0.7, \
			color='green',label='first time success')
	ax.bar(x_axis, ins_rest_distrib, \
			bottom=ins_distrib,width=-0.8, \
			align='edge', alpha=0.7, color='orange',label='restart but not success')
	ax.bar(x_axis, ins_surr_distrib, \
			bottom=ins_distrib+ins_rest_distrib,width=-0.8, \
			align='edge', alpha=0.7, color='red',label='surrender')
	ax.bar(x_axis, ins_rc1_distrib, \
			bottom=ins_distrib+ins_rest_distrib+ins_surr_distrib,\
			width=-0.8, align='edge', alpha=0.7, color='blue',label='success after 1 restart')
	ax.bar(x_axis, ins_rcm_distrib, \
			bottom=ins_distrib+ins_rest_distrib+ins_surr_distrib+ins_rc1_distrib,\
			width=-0.8, align='edge', alpha=0.4, color='blue',label='success after 2+ restart')
	ax.set_xlabel('human_time (second)',fontsize=14)
	# ax.set_ylabel('portion',fontsize=14)
	ax.set_ylim([0,1])
	plt.suptitle('Human_time distribution: ' + \
				instance + ', opt_len=' + str(opt_len),\
				fontsize=15)
	plt.title('mean=%.1f'%(sum(time_list) / float(len(time_list))) \
				+ ', median=%.1f'%median(time_list),\
				fontsize=15)
	fig_out_dir = out_dir + instance + '/' + instance + '_hum_time_distr.png'
	ax.grid(axis = 'y', alpha = 0.3)
	plt.legend(loc='upper right')
	plt.savefig(fig_out_dir)
	plt.close()
