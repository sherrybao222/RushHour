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
mag_depth_dict = {}
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
	cur_ins = all_instances[i]
	ins_dir = ins_file + all_instances[i] + '.json'
	my_car_list, my_red = MAG.json_to_car_list(ins_dir)
	my_board = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	mag_node_dict[cur_ins] = n_node
	mag_edge_dict[cur_ins] = n_edge
	countscc, _, maxlen = MAG.find_SCC(new_car_list)
	mag_countscc_dict[cur_ins] = countscc
	mag_maxscc_dict[cur_ins] = maxlen
	countc, _, maxc = MAG.find_cycles(new_car_list)
	mag_countcycle_dict[cur_ins] = countc
	mag_maxcycle_dict[cur_ins] = maxc
	depth, _ = MAG.longest_path(new_car_list)
	mag_depth_dict[cur_ins] = depth


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
y_depth = []
for i in range(len(all_instances)):
	y_human.append(mean_dict[all_instances[i]])
	y_opt.append(optimal_dict[all_instances[i]])
	y_nodes.append(mag_node_dict[all_instances[i]])
	y_edges.append(mag_edge_dict[all_instances[i]])
	y_countscc.append(mag_countscc_dict[all_instances[i]])
	y_maxscc.append(mag_maxscc_dict[all_instances[i]])
	y_countcycle.append(mag_countcycle_dict[all_instances[i]])
	y_maxcycle.append(mag_maxcycle_dict[all_instances[i]])
	y_depth.append(mag_depth_dict[all_instances[i]])
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
plt.plot(np.arange(len(all_instances)), y_countcycle, color='magenta', label='#cycle')
plt.plot(np.arange(len(all_instances)), y_maxcycle, color='brown', label='max cycle')
plt.plot(np.arange(len(all_instances)), y_depth, color='chartreuse', label='max depth')
# scatter1.set_zorder(2) # layer
# scatter2.set_zorder(2) 
# plt.plot(np.arange(len(all_instances)), y_nodes, color='blue', label='#nodes')
# plt.plot(np.arange(len(all_instances)), y_edges, color='red', label='#edges')
plt.title('Human Length, Optimal Length, MAG info')
plt.legend(loc='upper left')
# plt.show()
plt.savefig(out_dir)
plt.close()

# # plot nodes vs len, edges vs len
# fig, axarr = plt.subplots(2,2)
# axarr[0,0].scatter(y_nodes, y_human, color='orange')
# # axarr[0,0].set_xlabel('#nodes')
# axarr[0,0].set_ylabel('human_len')
# axarr[1,0].scatter(y_nodes, y_opt, color='green')
# axarr[1,0].set_xlabel('#nodes')
# axarr[1,0].set_ylabel('opt_len')
# axarr[0,1].scatter(y_edges, y_human, alpha=0.8, color='red')
# # axarr[0,1].set_xlabel('#edges')
# # axarr[0,1].set_ylabel('human_len')
# axarr[1,1].scatter(y_edges, y_opt, alpha=0.8, color='blue')
# axarr[1,1].set_xlabel('#edges')
# # axarr[1,1].set_ylabel('opt_len')
# axarr[0,0].set_title('new MAG #nodes vs len, #edges vs len', loc='left')
# #plt.show()
# plt.savefig(out_dir_1)
# plt.close()

# calculate pearson correlation and p-value
corr_list = []
p_list = []
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
corr, p = stats.spearmanr(y_human, y_depth)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & max depth: %s, P-value is %s\n" % (str(format(corr, '.6f')), str(format(p, '.6f'))))




# DEBUG:
# print(y_nodes)
# print(y_opt)
# print(stats.spearmanr(y_opt,y_nodes))
# a = np.random.rand(len(y_nodes))
# print(stats.spearmanr(np.array(y_opt), a))
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax1.scatter(y_opt+0.1*np.random.rand(len(y_nodes)),y_nodes+0.1*np.random.rand(len(y_nodes)))
# plt.show()


'''
results:

P-corr human_len & #nodes+#edges: 0.388062, P-value is 0.000902

P-corr opt_len & #nodes+#edges: 0.352236, P-value is 0.002786

P-corr human_len & #nodes: 0.418855, P-value is 0.000308

P-corr human_len & #edges: 0.358257, P-value is 0.002325

P-corr opt_len & #nodes: 0.355621, P-value is 0.002518

P-corr opt_len & #edges: 0.332458, P-value is 0.004925


SP-corr human_len & #nodes+#edges: 0.373598, P-value is 0.001444

SP-corr opt_len & #nodes+#edges: 0.302283, P-value is 0.010980

SP-corr human_len & #nodes: 0.475145, P-value is 0.000032

SP-corr human_len & #edges: 0.348554, P-value is 0.003106

SP-corr opt_len & #nodes: 0.352581, P-value is 0.002757

SP-corr opt_len & #edges: 0.286088, P-value is 0.016355
'''


