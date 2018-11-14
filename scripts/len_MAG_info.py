# mean human length, optimal length, new MAG info
import json, math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
from scipy import stats

len_file = '/Users/chloe/Documents/RushHour/data/paths.json'
ins_file = '/Users/chloe/Documents/RushHour/data/data_adopted/'
# sorted according to optimal length
# all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/len_MAG_info.png'
# corr_out_dir = '/Users/chloe/Documents/RushHour/data/len_MAG_corr.npy'
# p_out_dir = '/Users/chloe/Documents/RushHour/data/len_MAG_p.npy'
data = []
mag_node_dict = {}
mag_edge_dict = {}
mag_countscc_dict = {}
mag_maxscc_dict = {}
mag_countcycle_dict = {}
mag_maxcycle_dict = {}
mag_c_cycle_dict = {}
mag_depth_dict = {}
mag_ndepth_dict = {}
humanlen_dict = {}
mean_dict = {}
optimal_dict = {}
all_human_len = []
y_human_err = []

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
	if instance == 'prb29414':
		continue
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
	#print(all_human_len[i])
	#if np.std(all_human_len[i]) > 40:
	#	print(np.std(all_human_len[i]))

# process mag attributes
for i in range(len(all_instances)):
	# construct mag
	cur_ins = all_instances[i]
	ins_dir = ins_file + all_instances[i] + '.json'
	my_car_list, my_red = MAG.json_to_car_list(ins_dir)
	my_board = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	# num nodes, num edges
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	mag_node_dict[cur_ins] = n_node
	mag_edge_dict[cur_ins] = n_edge

	countscc, _, maxlen = MAG.find_SCC(new_car_list)
	mag_countscc_dict[cur_ins] = countscc
	mag_maxscc_dict[cur_ins] = maxlen

	countc, _, maxc = MAG.find_cycles(new_car_list)
	mag_countcycle_dict[cur_ins] = countc
	mag_maxcycle_dict[cur_ins] = maxc

	c_cycle = MAG.num_in_cycles(new_car_list)
	mag_c_cycle_dict[cur_ins] = c_cycle

	depth, paths = MAG.longest_path(new_car_list)
	mag_depth_dict[cur_ins] = depth
	mag_ndepth_dict[cur_ins] = len(paths)


# print special results
print('now print interesting results\n')
for i in range(len(all_instances)):
	ins = all_instances[i]
	m = mean_dict[ins]
	opt = optimal_dict[ins]
	n_nodes = mag_node_dict[ins]
	n_edge = mag_edge_dict[ins]
	if m >= opt * 3:
		print(ins + ' optimal = ' + str(opt) + ', human = ' + str(m) + ', #nodes = ' + str(n_nodes) + ', #edge = ' + str(n_edge))
# generate value lists
y_human = []
y_opt = []
y_nodes = []
y_edges = []
y_countscc = []
y_maxscc = []
y_countcycle = []
y_maxcycle = []
y_c_incycle = [] ##
y_depth = []
y_ndepth = [] ##
for i in range(len(all_instances)):
	y_human.append(mean_dict[all_instances[i]])
	y_opt.append(optimal_dict[all_instances[i]])
	y_nodes.append(mag_node_dict[all_instances[i]])
	y_edges.append(mag_edge_dict[all_instances[i]])
	y_countscc.append(mag_countscc_dict[all_instances[i]])
	y_maxscc.append(mag_maxscc_dict[all_instances[i]])
	y_c_incycle.append(mag_c_cycle_dict[all_instances[i]])
	y_countcycle.append(mag_countcycle_dict[all_instances[i]])
	y_maxcycle.append(mag_maxcycle_dict[all_instances[i]])
	y_depth.append(mag_depth_dict[all_instances[i]])
	y_ndepth.append(mag_ndepth_dict[all_instances[i]])
# generate figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.arange(len(all_instances)), y_human, alpha=0.9, color='orange', label='human')
ax.errorbar(np.arange(len(all_instances)), y_human, yerr=y_human_err, alpha=0.5, c='red', fmt='none')
ax.bar(np.arange(len(all_instances)), y_opt, alpha=0.65, color='green', label='optimal')
ax.set_xticklabels([])
ax.yaxis.set_major_locator(MaxNLocator(20))
ax.grid(axis = 'y', alpha = 0.3)
ax.set_facecolor('0.98')
ax.set_xlabel('Puzzles')
# ax.set_ylabel()
plt.plot(np.arange(len(all_instances)), y_nodes, color='blue', label='#node')
plt.plot(np.arange(len(all_instances)), y_edges, color='red', label='#edge')
plt.plot(np.arange(len(all_instances)), y_countscc, color='coral', label='#scc')
plt.plot(np.arange(len(all_instances)), y_maxscc, color='cyan', label='max scc')
plt.plot(np.arange(len(all_instances)), y_c_incycle, color='blueviolet', label='#cycles in cycle')
plt.plot(np.arange(len(all_instances)), y_countcycle, color='magenta', label='#cycle')
plt.plot(np.arange(len(all_instances)), y_maxcycle, color='brown', label='max cycle')
plt.plot(np.arange(len(all_instances)), y_depth, color='chartreuse', label='max depth')
plt.plot(np.arange(len(all_instances)), y_ndepth, color='olive', label='#longest paths')
# scatter1.set_zorder(2) # layer
# scatter2.set_zorder(2) 
# plt.plot(np.arange(len(all_instances)), y_nodes, color='blue', label='#nodes')
# plt.plot(np.arange(len(all_instances)), y_edges, color='red', label='#edges')
plt.title('Human Length, Optimal Length, MAG info')
plt.legend(loc='upper left')
# plt.show()
plt.savefig(out_dir)
plt.close()


# calculate pearson correlation and p-value
corr_list = []
p_list = []

corr, p = stats.spearmanr(y_human, y_opt)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & opt_len: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_nodes)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #nodes: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_edges)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #edges: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_countscc)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #scc: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_maxscc)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & maxscc: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_countcycle)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #cycle: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_maxcycle)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & max cycle: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_c_incycle)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #cycles in cycle: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_depth)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & max depth: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))

corr, p = stats.spearmanr(y_human, y_ndepth)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #longtest paths: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))


# create correlation plot
fig, axarr = plt.subplots(2, 5, figsize=(15, 6))

axarr[0,0].scatter(y_opt, y_human)
axarr[0,0].set_ylabel('opt_len')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[0],p_list[0])
axarr[0,0].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[0,1].scatter(y_nodes, y_human)
axarr[0,1].set_ylabel('#node')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[1],p_list[1])
axarr[0,1].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[0,2].scatter(y_edges, y_human)
axarr[0,2].set_ylabel('#edge')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[2],p_list[2])
axarr[0,2].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[0,3].scatter(y_countscc, y_human)
axarr[0,3].set_ylabel('#scc')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[3],p_list[3])
axarr[0,3].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[0,4].scatter(y_maxscc, y_human)
axarr[0,4].set_ylabel('max scc size')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[4],p_list[4])
axarr[0,4].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[1,0].scatter(y_countcycle, y_human)
axarr[1,0].set_ylabel('#cycle')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[5],p_list[5])
axarr[1,0].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[1,1].scatter(y_maxcycle, y_human)
axarr[1,1].set_ylabel('max cycle size')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[6],p_list[6])
axarr[1,1].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[1,2].scatter(y_c_incycle, y_human)
axarr[1,2].set_ylabel('#cycles in cycle')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[7],p_list[7])
axarr[1,2].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[1,3].scatter(y_depth, y_human)
axarr[1,3].set_ylabel('longtest path len from red')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[8],p_list[8])
axarr[1,3].set_title(t,y=0.85,fontsize=8,fontweight='bold')

axarr[1,4].scatter(y_ndepth, y_human)
axarr[1,4].set_ylabel('#longest paths')
t = 'Spearman corr: %1.5f,\nP-value=%1.5f'%(corr_list[9],p_list[9])
axarr[1,4].set_title(t,y=0.85,fontsize=8,fontweight='bold')

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1, rect=None) 
plt.suptitle('human_len vs MAG info', y=0.99, fontsize=13, fontweight='bold')
# plt.show()
plt.savefig('/Users/chloe/Documents/RushHour/figures/len_MAG_info_scatter.png')
plt.close()



