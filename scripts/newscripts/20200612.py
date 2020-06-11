'''
for Weiji's grant proposal June 12, 2020
0.5 margin on both sides left and right
Figure has 4 panels
Panel A: initial board visualization for example puzzle at 4 levels
Panel B: subjects distance to goal with actual puzzle (subject time course)
Panel C: MAG visualizations for initial board positions
Panle D: BFS tree visualization at the end of MakeMove call, with principal variation marked as blue
'''
import matplotlib.pyplot as plt
from Board import *
import pandas as pd
import random
from graphviz import Digraph


def plot_board(board, name='', show_car_label=False):
	''' 
	visualize board position and save into png figure
	figure name: lv10_<puzzlename>_board.png
	'''
	matrix = get_board_matrix(board)
	cmap = plt.cm.Set1
	cmap.set_bad(color='white')
	fig, ax = plt.subplots(figsize=(16,12))
	ax.set_xticks(np.arange(-0.5, 5, 1))
	ax.set_yticks(np.arange(-0.5, 5, 1))
	ax.set_axisbelow(True)
	ax.grid(b=True, which='major',color='gray', linestyle='-', linewidth=1, alpha=0.1)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.tick_params(axis='both', which='both',length=0)
	im = ax.imshow(matrix, cmap=cmap)
	if show_car_label:
		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				num = matrix[i, j]
				if num == 0:
					num = 'R'
				elif num > 0:
					num -= 1
				else:
					num = ''
				text = ax.text(j, i, num, ha="center", va="center", color="black", fontsize=36)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(4)
		ax.spines[axis].set_zorder(0)
	fig.patch.set_facecolor('white')
	fig.patch.set_alpha(0.2)
	plt.savefig('/Users/yichen/Desktop/'+name+'_board.png', 
			facecolor = fig.get_facecolor(), transparent = True) # lv10_<puzzlename>_board.png
	plt.close()
	return '/Users/yichen/Desktop/'+name+'_board.png'




def plot_MAG(board, name=''):
	matrix = get_board_matrix(board)
	cmap = plt.cm.Set1
	dot = Digraph(format='png')
	dot.node('r',label='R')
	mag = MAG(board)
	mag.construct()

	for f in finished_list:
		dot.node(f.tag)
		#print(f.tag)
		for edge_car in f.edge_to:
			dot.node(edge_car.tag)
			dot.edge(f.tag, edge_car.tag)
			#print("edge: " + edge_car.tag)
	# save and show 
	dot.render(filename, view=False)






if __name__ == '__main__':

	# data = pd.read_csv('/Users/yichen/Desktop/trialdata_valid_true_dist7_processed.csv')
	
	level6_puzzles = ['prb11647', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb8786', 'prb28111', 'prb32795', 'prb26567', 'prb21272', 'prb14047', 'prb14651', 'prb32695', 'prb13171', 'prb29232', 'prb15290', 'prb12604', 'prb20059']
	level10_puzzles = ['prb38526', 'prb3217', 'prb34092', 'prb29414', 'prb12715', 'prb62015', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb9718', 'prb14898', 'prb22436', 'prb62222', 'prb68910', 'prb33509', 'prb46224']
	level13_puzzles = ['prb38725', 'prb29585', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb47495', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb14485', 'prb34551', 'prb72800', 'prb65535']
	level15_puzzles = ['prb24227', 'prb45893', 'prb44171', 'prb25861', 'prb1267', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb29027', 'prb46639', 'prb46580', 'prb10166', 'prb24406', 'prb58853', 'prb57223']
	
	# for random_puzzle in level15_puzzles:
	# # random_puzzle = random.choice(level6_puzzles)
	# 	print('Random chosen puzzle: '+random_puzzle)
	# 	board = json_to_board('/Users/yichen/Documents/RushHour/exp_data/data_adopted/'+random_puzzle+'.json')
	# 	plot_board(board, name='lv15_'+random_puzzle)








