'''
Visualize the goodness of model fit.
Using summary statistics:
visualze the MAG attribute of moves given by model prediction vs human decision.
py27
'''
import MAG, rushhour
import random, sys, copy, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from operator import methodcaller
import my_llfast

# load data
# [-0.5391 -3.9844 3.8281 -5.5469 -2.5781 -8.8867 -2.5469 -0.7656 1.4844 5.3320]
model_params = [-0.5391, -3.9844, 3.8281, -5.5469, -2.5781, -8.8867, -2.5469, -0.7656, 1.4844, 5.3320]
human_nodes = pickle.load(open('/Users/chloe/Documents/RushHour/scripts/node_list.pickle', 'r'))
# MAG attribute to be plotted
model_mag1 = [] # number of edges in the MAG
model_mag2 = [] # number of nodes in the MAG
human_mag1 = []
human_mag2 = []
model_mag1.append(0) # padding
model_mag2.append(0)

for human_node in human_nodes[:-1]:
	# print('-------------------------------------------------')
	model_node = my_llfast.wrap_make_move(model_params, human_node)
	b, r = MAG.construct_board(model_node.get_carlist())
	fl = MAG.construct_mag(b, r)
	_ , e = MAG.get_mag_attr(fl)
	model_mag1.append(e)
	cl = MAG.assign_level(fl, r)
	n = np.sum(MAG.get_num_cars_from_levels(cl, len(model_params)-2))
	model_mag2.append(n)
	# print('model')
	# print(model_node.board_to_str())
	b, r = MAG.construct_board(human_node.get_carlist())
	fl = MAG.construct_mag(b, r)
	_ , e = MAG.get_mag_attr(fl)
	human_mag1.append(e)
	cl = MAG.assign_level(fl, r)
	n = np.sum(MAG.get_num_cars_from_levels(cl, len(model_params)-2))
	human_mag2.append(n)
	# print('human')
	# print(human_node.board_to_str())
b, r = MAG.construct_board(human_node.get_carlist())
fl = MAG.construct_mag(b, r)
n, e = MAG.get_mag_attr(fl)
human_mag1.append(e)
cl = MAG.assign_level(fl, r)
n = np.sum(MAG.get_num_cars_from_levels(cl, len(model_params)-2))
human_mag2.append(n)

print('Number of edges')
print(human_mag1)
print(model_mag1)
print('Number of nodes')
print(human_mag2)
print(model_mag2)
# sys.exit()
plt.plot(np.arange(len(human_nodes)-1), human_mag2[1:], color='green')
plt.plot(np.arange(len(human_nodes)-1), model_mag2[1:], color='orange')
plt.show()
plt.close()