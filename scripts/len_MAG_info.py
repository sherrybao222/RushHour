# puzzle level (initial state): mean human length, optimal length, new MAG info 
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
data_out = '/Users/chloe/Documents/RushHour/exp_data/MAG_info/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
bar_out_dir = '/Users/chloe/Documents/RushHour/puzzle_figures/len_MAG_info.png'
scatter_out = '/Users/chloe/Documents/RushHour/puzzle_figures/len_MAG_info_scatter.png'
data = []
mag_node_dict = {} # number of nodes
mag_edge_dict = {} # number of edges
mag_en_dict = {} # #edges/#nodes
mag_enp_dict = {} # #edges/(#nodes-#leaf)
mag_e2n_dict = {} # #edges^2 / #nodes
mag_countscc_dict = {} # number of SCC
mag_maxscc_dict = {} # max SCC size
mag_countcycle_dict = {} # number of cycles
mag_maxcycle_dict = {} # max cycle size
mag_c_cycle_dict = {} # number of cycles in cycles
mag_nnc_dict = {} # number of nodes in cycles
mag_pnc_dict = {} # proportion of nodes in cycles
mag_depth_dict = {} # longest path len from red
mag_ndepth_dict = {} # number of longest paths from red
mag_gcluster_dict = {} # global clustering coefficient
mag_lcluster_dict = {} # mean average clustering coefficient

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
	new_car_list = MAG.construct_mag(my_board, my_red)
	
	# mag_node_dict = {} # number of nodes
	# mag_edge_dict = {} # number of edges
	# mag_en_dict = {} # #edges/#nodes
	# mag_enp_dict = {} # #edges/(#nodes-#leaf)
	# mag_e2n_dict = {} # #edges^2 / #nodes
	# mag_countscc_dict = {} # number of SCC
	# mag_maxscc_dict = {} # max SCC size
	# mag_countcycle_dict = {} # number of cycles
	# mag_maxcycle_dict = {} # max cycle size
	# mag_c_cycle_dict = {} # number of cycles in cycles
	# mag_nnc_dict = {} # number of nodes in cycles
	# mag_pnc_dict = {} # proportion of nodes in cycles
	# mag_depth_dict = {} # longest path len from red
	# mag_ndepth_dict = {} # number of longest paths from red
	# mag_gcluster_dict = {} # global clustering coefficient
	# mag_lcluster_dict = {} # mean average clustering coefficient

	# num nodes, num edges
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	mag_node_dict[cur_ins] = n_node
	mag_edge_dict[cur_ins] = n_edge
	# edge nodes ratio, branching factor
	ebn = MAG.e_by_n(new_car_list)
	ebpn = MAG.e_by_pn(new_car_list)
	e2n = MAG.e2_by_n(new_car_list)
	mag_en_dict[cur_ins] = ebn
	mag_enp_dict[cur_ins] = ebpn
	mag_e2n_dict[cur_ins] = e2n
	# SCC factors
	countscc, _, maxlen = MAG.find_SCC(new_car_list)
	mag_countscc_dict[cur_ins] = countscc
	mag_maxscc_dict[cur_ins] = maxlen
	# cycle information
	# cycle count, max cycle size
	countc, _, maxc = MAG.find_cycles(new_car_list) 
	mag_countcycle_dict[cur_ins] = countc
	mag_maxcycle_dict[cur_ins] = maxc
	# number of cycles in cycles
	c_cycle = MAG.num_in_cycles(new_car_list)
	mag_c_cycle_dict[cur_ins] = c_cycle
	# number of nodes in cycles
	n_nc, _ = MAG.num_nodes_in_cycle(new_car_list)
	mag_nnc_dict[cur_ins] = n_nc
	# proportion of nodes in cycles
	pro = MAG.pro_nodes_in_cycle(new_car_list)
	mag_pnc_dict[cur_ins] = pro
	# longest path len from red and number of longest paths
	depth, paths = MAG.longest_path(new_car_list)
	mag_depth_dict[cur_ins] = depth
	mag_ndepth_dict[cur_ins] = len(paths)
	# clustering coefficient
	gcluster = MAG.global_cluster_coef(new_car_list)
	lcluster = MAG.av_local_cluster_coef(new_car_list)
	mag_gcluster_dict[cur_ins] = gcluster
	mag_lcluster_dict[cur_ins] = lcluster


# generate value lists
y_human = []
y_opt = []
y_nodes = []
y_edges = []
y_en = []
y_enp = []
y_e2n = []
y_countscc = []
y_maxscc = []
y_countcycle = []
y_maxcycle = []
y_c_incycle = []
y_nnc = []
y_pnc = []
y_depth = []
y_ndepth = []
y_gcluster = []
y_lcluster = []

# mag_node_dict = {} # number of nodes
# mag_edge_dict = {} # number of edges
# mag_en_dict = {} # #edges/#nodes
# mag_enp_dict = {} # #edges/(#nodes-#leaf)
# mag_e2n_dict = {} # #edges^2 / #nodes
# mag_countscc_dict = {} # number of SCC
# mag_maxscc_dict = {} # max SCC size
# mag_countcycle_dict = {} # number of cycles
# mag_maxcycle_dict = {} # max cycle size
# mag_c_cycle_dict = {} # number of cycles in cycles
# mag_nnc_dict = {} # number of nodes in cycles
# mag_pnc_dict = {} # proportion of nodes in cycles
# mag_depth_dict = {} # longest path len from red
# mag_ndepth_dict = {} # number of longest paths from red
# mag_gcluster_dict = {} # global clustering coefficient
# mag_lcluster_dict = {} # mean average clustering coefficient

for i in range(len(all_instances)):
	y_human.append(mean_dict[all_instances[i]])
	y_opt.append(optimal_dict[all_instances[i]])
	y_nodes.append(mag_node_dict[all_instances[i]])
	y_edges.append(mag_edge_dict[all_instances[i]])
	y_en.append(mag_en_dict[all_instances[i]])
	y_enp.append(mag_enp_dict[all_instances[i]])
	y_e2n.append(mag_e2n_dict[all_instances[i]])
	y_countscc.append(mag_countscc_dict[all_instances[i]])
	y_maxscc.append(mag_maxscc_dict[all_instances[i]])
	y_countcycle.append(mag_countcycle_dict[all_instances[i]])
	y_maxcycle.append(mag_maxcycle_dict[all_instances[i]])
	y_c_incycle.append(mag_c_cycle_dict[all_instances[i]])
	y_nnc.append(mag_nnc_dict[all_instances[i]])
	y_pnc.append(mag_pnc_dict[all_instances[i]])
	y_depth.append(mag_depth_dict[all_instances[i]])
	y_ndepth.append(mag_ndepth_dict[all_instances[i]])
	y_gcluster.append(mag_gcluster_dict[all_instances[i]])
	y_lcluster.append(mag_lcluster_dict[all_instances[i]])

# bar plots human_len, opt_len, and MAG curves
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
# mag_node_dict = {} # number of nodes
# mag_edge_dict = {} # number of edges
# mag_en_dict = {} # #edges/#nodes
# mag_enp_dict = {} # #edges/(#nodes-#leaf)
# mag_e2n_dict = {} # #edges^2 / #nodes
# mag_countscc_dict = {} # number of SCC
# mag_maxscc_dict = {} # max SCC size
# mag_countcycle_dict = {} # number of cycles
# mag_maxcycle_dict = {} # max cycle size
# mag_c_cycle_dict = {} # number of cycles in cycles
# mag_nnc_dict = {} # number of nodes in cycles
# mag_pnc_dict = {} # proportion of nodes in cycles
# mag_depth_dict = {} # longest path len from red
# mag_ndepth_dict = {} # number of longest paths from red
# mag_gcluster_dict = {} # global clustering coefficient
# mag_lcluster_dict = {} # mean average clustering coefficient
plt.plot(np.arange(len(all_instances)), y_nodes, color='lightcoral', label='#node')
plt.plot(np.arange(len(all_instances)), y_edges, color='salmon', label='#edge')
plt.plot(np.arange(len(all_instances)), y_en, color='firebrick', label='#edge/#node')
plt.plot(np.arange(len(all_instances)), y_enp, color='maroon', label='#edge/(#node-#leaf)')
plt.plot(np.arange(len(all_instances)), y_e2n, color='firebrick', label='#edge^2/#node')
plt.plot(np.arange(len(all_instances)), y_countscc, color='gold', label='#scc')
plt.plot(np.arange(len(all_instances)), y_maxscc, color='khaki', label='max scc')
plt.plot(np.arange(len(all_instances)), y_countcycle, color='olivedrab', label='#cycle')
plt.plot(np.arange(len(all_instances)), y_maxcycle, color='lime', label='max cycle')
plt.plot(np.arange(len(all_instances)), y_c_incycle, color='forestgreen', label='#cycles in cycle')
plt.plot(np.arange(len(all_instances)), y_nnc, color='cyan', label='#nodes in cycle')
plt.plot(np.arange(len(all_instances)), y_pnc, color='skyblue', label='pro nodes in cycle')
plt.plot(np.arange(len(all_instances)), y_depth, color='blueviolet', label='max depth')
plt.plot(np.arange(len(all_instances)), y_ndepth, color='purple', label='#longest paths')
plt.plot(np.arange(len(all_instances)), y_gcluster, color='green', label='glb cluster coef')
plt.plot(np.arange(len(all_instances)), y_lcluster, color='green', label='mean local cluster coef')
plt.title('Human Length, Optimal Length, MAG info')
plt.legend(loc='upper left')
# plt.show()
plt.savefig(bar_out_dir)
plt.close()

# save data
np.save(data_out + 'y_human.npy',y_human) # mean human len
np.save(data_out + 'y_opt.npy', y_opt) # opt len
np.save(data_out + 'y_nodes.npy',y_nodes) # num of nodes
np.save(data_out + 'y_edges.npy',y_edges) # num of edges
np.save(data_out + 'y_en.npy',y_en) # num edge/num node
np.save(data_out + 'y_enp.npy',y_enp) # num edge/(num node - num leaf)
np.save(data_out + 'y_e2n.npy',y_e2n) # num edge sq / num node
np.save(data_out + 'y_countscc.npy',y_countscc) # num of scc
np.save(data_out + 'y_maxscc.npy',y_maxscc) # max scc size
np.save(data_out + 'y_countcycle.npy',y_countcycle) # num of cycles
np.save(data_out + 'y_maxcycle.npy',y_maxcycle) # max cycle size
np.save(data_out + 'y_c_incycle.npy',y_c_incycle) # num of cycles in cycles
np.save(data_out + 'y_nnc.npy',y_nnc) # num of nodes in cycles
np.save(data_out + 'y_pnc.npy',y_pnc) # proportion of nodes in cycles
np.save(data_out + 'y_depth.npy',y_depth) # len of longest path from red
np.save(data_out + 'y_ndepth.npy',y_ndepth) # num of longest paths from red
np.save(data_out + 'y_gcluster.npy',y_gcluster) # global clustering coefficient
np.save(data_out + 'y_lcluster.npy',y_lcluster) # mean local clustering coefficient

# calculate pearson correlation and p-value for human_len and MAG info
corr_list = []
p_list = []
# mag_node_dict = {} # number of nodes
# mag_edge_dict = {} # number of edges
# mag_en_dict = {} # #edges/#nodes
# mag_enp_dict = {} # #edges/(#nodes-#leaf)
# mag_e2n_dict = {} # #edges^2 / #nodes
# mag_countscc_dict = {} # number of SCC
# mag_maxscc_dict = {} # max SCC size
# mag_countcycle_dict = {} # number of cycles
# mag_maxcycle_dict = {} # max cycle size
# mag_c_cycle_dict = {} # number of cycles in cycles
# mag_nnc_dict = {} # number of nodes in cycles
# mag_pnc_dict = {} # proportion of nodes in cycles
# mag_depth_dict = {} # longest path len from red
# mag_ndepth_dict = {} # number of longest paths from red
# mag_gcluster_dict = {} # global clustering coefficient
# mag_lcluster_dict = {} # mean average clustering coefficient
corr, p = stats.pearsonr(y_human, y_opt)
print("PearsonR human_len & opt_len: %s, P-value is %s\n" % (str(format(corr, '.3g')), str(format(p, '.2g'))))

# corr, p = stats.spearmanr(y_human, y_opt)
# print("SP-R human_len & opt_len: %s, P-value is %s\n" % (str(format(corr, '.3g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_nodes)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #node: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_edges)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #edge: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_en)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #edge/#node: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_enp)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #edge/(#node-#leaf): %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_e2n)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #edge^2/#node: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_countscc)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #scc: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_maxscc)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & maxscc: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_countcycle)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #cycle: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_maxcycle)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & max cycle: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_c_incycle)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #cycles in cycle: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_nnc)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #nodes in cycles: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_pnc)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & proportion nodes in cycle: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_depth)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & max depth: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_ndepth)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & #longtest paths: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_gcluster)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & global cluster coef: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))

corr, p = stats.spearmanr(y_human, y_lcluster)
corr_list.append(corr)
p_list.append(p)
print("SP-corr human_len & mean loc cluster coef: %s, P-value is %s\n" % (str(format(corr, '.2g')), str(format(p, '.2g'))))


# create scatter plot to show correlation and p
fig, axarr = plt.subplots(4, 4, figsize=(30, 30))
# mag_node_dict = {} # number of nodes
# mag_edge_dict = {} # number of edges
# mag_en_dict = {} # #edges/#nodes
# mag_enp_dict = {} # #edges/(#nodes-#leaf)
# mag_e2n_dict = {} # #edges^2 / #nodes
# mag_countscc_dict = {} # number of SCC
# mag_maxscc_dict = {} # max SCC size
# mag_countcycle_dict = {} # number of cycles
# mag_maxcycle_dict = {} # max cycle size
# mag_c_cycle_dict = {} # number of cycles in cycles
# mag_nnc_dict = {} # number of nodes in cycles
# mag_pnc_dict = {} # proportion of nodes in cycles
# mag_depth_dict = {} # longest path len from red
# mag_ndepth_dict = {} # number of longest paths from red
# mag_gcluster_dict = {} # global clustering coefficient
# mag_lcluster_dict = {} # mean average clustering coefficient
axarr[0,0].scatter(y_human, y_nodes)
axarr[0,0].set_ylabel('#node', fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[0], '.3f'), format(p_list[0], '.2g'))
axarr[0,0].set_title(t,y=0.85,fontsize=17)
axarr[0,0].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[0,0].tick_params(axis='both', labelsize=17)

axarr[0,1].scatter(y_human, y_edges)
axarr[0,1].set_ylabel('#edge',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[1], '.3f'), format(p_list[1], '.2g'))
axarr[0,1].set_title(t,y=0.85,fontsize=17)
axarr[0,1].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[0,1].tick_params(axis='both', labelsize=17)

axarr[0,2].scatter(y_human, y_en)
axarr[0,2].set_ylabel('#edge/#node',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[2], '.3f'), format(p_list[2], '.2g'))
axarr[0,2].set_title(t,y=0.85,fontsize=17)
axarr[0,2].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=False))
axarr[0,2].tick_params(axis='both', labelsize=17)

axarr[0,3].scatter(y_human, y_enp)
axarr[0,3].set_ylabel('#edge/(#node-#leaf)',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[3], '.3f'), format(p_list[3], '.2g'))
axarr[0,3].set_title(t,y=0.85,fontsize=17)
axarr[0,3].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=False))
axarr[0,3].tick_params(axis='both', labelsize=17)


axarr[1,0].scatter(y_human, y_e2n)
axarr[1,0].set_ylabel('#edge^2/#node',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[4], '.3f'), format(p_list[4], '.2g'))
axarr[1,0].set_title(t,y=0.85,fontsize=17)
axarr[1,0].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=False))
axarr[1,0].tick_params(axis='both', labelsize=17)

axarr[1,1].scatter(y_human, y_countscc)
axarr[1,1].set_ylabel('#SCC',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[5], '.3f'), format(p_list[5], '.2g'))
axarr[1,1].set_title(t,y=0.85,fontsize=17)
axarr[1,1].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[1,1].tick_params(axis='both', labelsize=17)

axarr[1,2].scatter(y_human, y_maxscc)
axarr[1,2].set_ylabel('max SCC size',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[6], '.3f'), format(p_list[6], '.2g'))
axarr[1,2].set_title(t,y=0.85,fontsize=17)
axarr[1,2].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[1,2].tick_params(axis='both', labelsize=17)

axarr[1,3].scatter(y_human, y_countcycle)
axarr[1,3].set_ylabel('#cycles', fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[7], '.3f'), format(p_list[7], '.2g'))
axarr[1,3].set_title(t,y=0.85,fontsize=17)
axarr[1,3].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[1,3].tick_params(axis='both', labelsize=17)

axarr[2,0].scatter(y_human, y_maxcycle)
axarr[2,0].set_ylabel('max cycle size',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[8], '.3f'), format(p_list[8], '.2g'))
axarr[2,0].set_title(t,y=0.85,fontsize=17)
axarr[2,0].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[2,0].tick_params(axis='both', labelsize=17)

axarr[2,1].scatter(y_human, y_c_incycle)
axarr[2,1].set_ylabel('#cycles in cycles',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[9], '.3f'), format(p_list[9], '.2g'))
axarr[2,1].set_title(t,y=0.85,fontsize=17)
axarr[2,1].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[2,1].tick_params(axis='both', labelsize=17)

axarr[2,2].scatter(y_human, y_nnc)
axarr[2,2].set_ylabel('#node in cycles',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[10], '.3f'), format(p_list[10], '.2g'))
axarr[2,2].set_title(t,y=0.85,fontsize=17)
axarr[2,2].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[2,2].tick_params(axis='both', labelsize=17)

axarr[2,3].scatter(y_human, y_pnc)
axarr[2,3].set_ylabel('proportion nodes in cycles',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[11], '.3f'), format(p_list[11], '.2g'))
axarr[2,3].set_title(t,y=0.85,fontsize=17)
axarr[2,3].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=False))
axarr[2,3].tick_params(axis='both', labelsize=17)

axarr[3,0].scatter(y_human, y_depth)
axarr[3,0].set_ylabel('longest path len from red',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[12], '.3f'), format(p_list[12], '.2g'))
axarr[3,0].set_title(t,y=0.85,fontsize=17)
axarr[3,0].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[3,0].tick_params(axis='both', labelsize=17)

axarr[3,1].scatter(y_human, y_ndepth)
axarr[3,1].set_ylabel('#longest paths from red',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[13], '.3f'), format(p_list[13], '.2g'))
axarr[3,1].set_title(t,y=0.85,fontsize=17)
axarr[3,1].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
axarr[3,1].tick_params(axis='both', labelsize=17)

axarr[3,2].scatter(y_human, y_gcluster)
axarr[3,2].set_ylabel('glb cluster coef',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[14], '.3f'), format(p_list[14], '.2g'))
axarr[3,2].set_title(t,y=0.85,fontsize=17)
axarr[3,2].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=False))
axarr[3,2].tick_params(axis='both', labelsize=17)

axarr[3,3].scatter(y_human, y_lcluster)
axarr[3,3].set_ylabel('mean loc cluster coef',fontsize=20, fontweight='bold')
t = 'rho=%s, p=%s'%(format(corr_list[15], '.3f'), format(p_list[15], '.2g'))
axarr[3,3].set_title(t,y=0.85,fontsize=17)
axarr[3,3].yaxis.set_major_locator(MaxNLocator(nbins=6,integer=False))
axarr[3,3].tick_params(axis='both', labelsize=17)

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1, rect=None) 
plt.suptitle('human_len vs MAG info', y=0.999, fontsize=22, fontweight='bold')
# plt.show()
plt.savefig(scatter_out)
plt.close()



