# mean human length, optimal length, new MAG attributes
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG

len_file = '/Users/chloe/Documents/RushHour/data/paths.json'
ins_file = '/Users/chloe/Documents/RushHour/data/data_adopted/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/len_MAG.png'
data = []
mag_node_dict = {}
mag_edge_dict = {}
instance_dict = {}
mean_dict = {}
optimal_dict = {}

# preprocess human & optimal len dict
for i in range(len(all_instances)):
	instance_dict[all_instances[i] + '_count'] = 0
	instance_dict[all_instances[i]+'_len'] = 0
	mean_dict[all_instances[i]] = 0
	optimal_dict[all_instances[i]] = 0
with open(len_file) as f:
	for line in f:
		data.append(json.loads(line))
for i in range(0, len(data)):
	line = data[i]
	instance = line['instance']
	complete = line['complete']
	human_len = int(line['human_length'])
	opt_len = int(line['optimal_length'])
	if complete == 'False':
		continue
	else:
		instance_dict[instance + '_count'] += 1
		instance_dict[instance + '_len'] = instance_dict[instance + '_len'] + human_len
		optimal_dict[instance] = opt_len
for i in range(len(all_instances)): # calculate mean human len
	if instance_dict[all_instances[i] + '_count'] == 0:
		mean_dict[all_instances[i]] = 0
		continue
	else:
		mean_dict[all_instances[i]] = instance_dict[all_instances[i] + '_len'] / instance_dict[all_instances[i] + '_count']

# process mag attributes
for i in range(len(all_instances)):
	cur_ins = all_instances[i]
	ins_dir = ins_file + all_instances[i] + '.json'
	my_car_list, my_red = MAG.json_to_car_list(ins_dir)
	my_board = MAG.construct_board(my_car_list)
	new_car_list = MAG.construct_mag(my_board, my_red)
	#MAG.visualize_mag(new_car_list, "/Users/chloe/Desktop/test_mag.gv")
	n_node, n_edge = MAG.get_mag_attr(new_car_list)
	#print("num_node: " + str(n_node))
	#print("num_edge: " + str(n_edge))
	mag_node_dict[cur_ins] = n_node
	mag_edge_dict[cur_ins] = n_edge
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
for i in range(len(all_instances)):
	y_human.append(mean_dict[all_instances[i]])
	y_opt.append(optimal_dict[all_instances[i]])
	y_nodes.append(mag_node_dict[all_instances[i]])
	y_edges.append(mag_edge_dict[all_instances[i]])
# generate figure
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(np.arange(len(all_instances)), y_human, alpha=0.9, color='orange', label='human')
rect = ax.bar(np.arange(len(all_instances)), y_opt, alpha=0.65, color='green', label='optimal')
ax.set_xticklabels([])
ax.yaxis.set_major_locator(MaxNLocator(20))
ax.grid(axis = 'y', alpha = 0.3)
ax.set_facecolor('0.98')
ax.set_xlabel('Puzzles')
ax.set_ylabel('#moves, #nodes, #edge')
plt.plot(np.arange(len(all_instances)), y_nodes, color='blue', label='#nodes')
plt.plot(np.arange(len(all_instances)), y_edges, color='red', label='#edges')
plt.title('Human Length, Optimal Length, # MAG Nodes, # MAG Edges')
plt.legend(loc='upper left')
#plt.show()
plt.savefig(out_dir)
plt.close()

# calculate pearson correlation
print("P-corr human_len & #nodes+#edges: \n" + str(np.corrcoef(y_human, np.array(y_nodes)+np.array(y_edges))))
print("P-corr opt_len & #nodes+#edges: \n" + str(np.corrcoef(y_opt, np.array(y_nodes)+np.array(y_edges))))
print("P-corr human_len & #nodes: \n" + str(np.corrcoef(y_human, y_nodes)))
print("P-corr human_len & #edges: \n" + str(np.corrcoef(y_human, y_edges)))
print("P-corr opt_len & #nodes: \n" + str(np.corrcoef(y_opt, y_nodes)))
print("P-corr opt_len & #edges: \n" + str(np.corrcoef(y_opt, y_edges)))







