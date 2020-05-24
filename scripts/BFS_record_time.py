''' 
measure time comlexity
BFS model self-defined ll function,
speeded version, prepared for BADS in MATLAB,
python3 or py27
'''
import random, copy, pickle, os, sys, time
from operator import attrgetter
import multiprocessing as mp
import numpy as np
from numpy import recfromcsv
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import scipy.stats as stats
from sklearn.model_selection import KFold
from Car import *
from Board import *
from Node import *

######################################## BFS MODEL ##############

class Params:
	def __init__(self, w1, w2, w3, w4, w5, w6, w7, 
					stopping_probability,
					pruning_threshold,
					lapse_rate,
					feature_dropping_rate=0.0,
					mu=0.0, sigma=1.0):
		self.w0 = 0.0
		self.w1 = w1
		self.w2 = w2
		self.w3 = w3
		self.w4 = w4
		self.w5 = w5
		self.w6 = w6
		self.w7 = w7
		self.weights = [self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7]
		self.num_weights = len(self.weights)
		self.mu = mu
		self.sigma = sigma
		self.feature_dropping_rate = feature_dropping_rate
		self.stopping_probability = stopping_probability
		self.pruning_threshold = pruning_threshold
		self.lapse_rate = lapse_rate

def DropFeatures(probability):
	pass

def Lapse(probability):
	''' return true with a probability '''
	return random.random() < probability

def Stop(probability): 
	''' return true with a probability '''
	return random.random() < probability

def RandomMove(node, params):
	''' make a random move and return the resulted node '''
	assert not is_solved(node.board, node.red), "RandomMove input node is already solved."
	InitializeChildren(node, params)
	return random.choice(node.children)
	
def InitializeChildren(node, params):
	''' 
		initialize the list of children nodes
		(using all legal moves) 
	'''
	InitializeChildren_start = time.time()
	
	# find all legal moves
	all_moves = all_legal_moves(node.car_list, node.board)
	# print('len(all_legal_moves)='+str(len(all_moves)))
	time_dict['all_legal_moves'].append(time.time() - InitializeChildren_start)
	
	for (tag, pos) in all_moves:

		move_xy_start = time.time()
		# move board position to simulate possible move 
		new_list, _ = move_xy(node.car_list, tag, pos[0], pos[1])
		time_dict['move_xy'].append(time.time() - move_xy_start)

		childNode_start = time.time()	
		# create child node
		child = Node(new_list, params)
		child.parent = node
		node.children.append(child)
		time_dict['create_Node'].append(time.time() - childNode_start)
		time_dict['Node_board_time'].append(child.board_time)
		time_dict['Node_find_red_time'].append(child.find_red_time)
		time_dict['Node_construct_mag_time'].append(child.construct_mag_time)
		time_dict['Node_assign_level_time'].append(child.assign_level_time)
		time_dict['Node_value_time'].append(child.value_time)

	time_dict['InitializeChildren'].append(time.time() - InitializeChildren_start)

def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	while len(n.children) != 0:
		n = ArgmaxChild(n)
	return n, is_solved(n.board, n.red)
 
def ExpandNode(node, params):
	''' 
	create all possible nodes under input node, 
	cut the ones below threshold 
	'''
	InitializeChildren(node, params)
	Vmaxchild = ArgmaxChild(node)
	for child_idx in range(len(node.children))[::-1]: # iterate in reverse order
		if abs(node.children[child_idx].value - Vmaxchild.value) > params.pruning_threshold:
			node.remove_child(child_idx)

def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	this_node.value = ArgmaxChild(this_node).value
	if this_node != root_node:
		Backpropagate(this_node.parent, root_node)

def ArgmaxChild(node): 
	''' 
		return the child with max value 
	'''
	ArgmaxChild_start = time.time()
	result = max(node.children, key=attrgetter('value'))
	time_dict['ArgmaxChild'].append(time.time() - ArgmaxChild_start)
	return result
	

def MakeMove(root, params, hit=False, verbose=False):
	''' 
	`	returns an optimal move to make next 
		according to value function and current board position
	'''
	MakeMove_start = time.time()
	beforewhile_start = time.time()
	# print('[[[[[[[[[[ MakeMove start')
	if hit: # for ibs, if matches human decision, return root node
		time_dict['MakeMove'].append(time.time()- MakeMove_start)
		return root
	assert len(root.children) == 0
	# check if root node is already a winning position
	if is_solved(root.board, root.red):
		# move red car to position 16 if rootis a winning position
		new_carlist, _ = move(root.car_list, 'r', 16)
		result = Node(new_carlist, params)
		return result
	if Lapse(params.lapse_rate): # random move
		RandomMove_start = time.time()
		result = RandomMove(root, params)
		time_dict['beforewhile'].append(time.time() - beforewhile_start)
		time_dict['MakeMove'].append(time.time()- MakeMove_start)
		return result
	else:
		DropFeatures(params.feature_dropping_rate)
		Stop_start = time.time()
		should_stop = Stop(probability=params.stopping_probability)
		time_dict['Stop'].append(time.time() - Stop_start)
		time_dict['beforewhile'].append(time.time() - beforewhile_start)
		while_start = time.time()
		while not should_stop:
			SelectNode_start = time.time()
			leaf, leaf_is_solution = SelectNode(root)
			time_dict['SelectNode'].append(time.time()- SelectNode_start)
			if leaf_is_solution:
				Backpropagate_start = time.time()
				Backpropagate(leaf.parent, root)
				time_dict['Backpropagate'].append(time.time() - Backpropagate_start)
				Stop_start = time.time()
				should_stop = Stop(probability=params.stopping_probability)
				time_dict['Stop'].append(time.time() - Stop_start)
				continue
			ExpandNode_start = time.time()
			ExpandNode(leaf, params)
			time_dict['ExpandNode'].append(time.time() - ExpandNode_start)
			Backpropagate_start = time.time()
			Backpropagate(leaf, root)
			time_dict['Backpropagate'].append(time.time() - Backpropagate_start)
			Stop_start = time.time()
			should_stop = Stop(probability=params.stopping_probability)
			time_dict['Stop'].append(time.time() - Stop_start)
		time_dict['insidewhile'].append(time.time() - while_start)
		afterwhile_start = time.time()
	if root.children == []: # if did not enter while loop at all
		ExpandNode_start = time.time()
		ExpandNode(root, params)
		time_dict['ExpandNode'].append(time.time() - ExpandNode_start)
	result = ArgmaxChild(root)
	time_dict['afterwhile'].append(time.time() - afterwhile_start)
	time_dict['MakeMove'].append(time.time()- MakeMove_start)
	# print('end MakeMove ]]]]]]]')
	return result


if __name__ == '__main__':
	# data path definitions
	all_datapath = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
	outputpath = '/Users/chloe/Desktop/timedict.csv'
	instancepath='/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	subject_path = '/Users/chloe/Desktop/subjects/' + random.choice(os.listdir('/Users/chloe/Desktop/subjects/'))
	df = pd.read_csv(subject_path)
	print('subject '+str(subject_path)+'\nsample size: '+str(df.shape[0]))

	# initialize parameters
	inparams = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 0.01]
	all_time_dict = {'MakeMove':[], 
					'insidewhile':[], 'beforewhile':[], 'afterwhile':[],
					'Stop':[], 'SelectNode':[], 'Backpropagate':[], 'ArgmaxChild':[],
					'ExpandNode':[], 
						'InitializeChildren':[], 
							'all_legal_moves':[], 'move_xy':[], 
							'create_Node': [], 
								'Node_board_time':[], 'Node_find_red_time':[], 'Node_construct_mag_time':[], 'Node_assign_level_time':[], 'Node_value_time':[],								
					}

	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
					w4=inparams[3], w5=inparams[4], w6=inparams[5], 
					w7=inparams[6], 
					stopping_probability=inparams[7],
					pruning_threshold=inparams[8],
					lapse_rate=inparams[9])
	
	# iterate over each move
	for idx, row in df.iterrows():
		print(idx, subject_path)
		# initialize record for current move
		time_dict = {}
		for key in all_time_dict.keys():
			time_dict[key] = []
		# read current move data 
		if row['event'] == 'start':
			# load new instance data
			instance = row['instance']
			ins_file = instancepath+instance+'.json'
			cur_carlist = json_to_car_list(ins_file)
			continue
		if row['piece']=='r' and row['move']==16 and row['optlen']==1: # win
			# skip the winning moves
			continue

		# call makemove and record time data
		MakeMove(Node(cur_carlist, params), params)
		
		# change the board and perform the current move
		piece = str(row['piece'])
		move_to = int(row['move'])
		cur_carlist, _ = move(cur_carlist, piece, move_to)
		
		# summarize all records from current move
		for key in all_time_dict.keys():
			all_time_dict[key].append(sum(time_dict[key]))

	# all moves completed, convert to dataframe
	all_time_df = pd.DataFrame.from_dict(all_time_dict)
	# save files as csv and calculate summary statistics
	all_time_df.to_csv(outputpath, index = False, header=True)
	# calculate proportion wrt MakeMove, except MakeMove itself
	df = pd.read_csv(outputpath)
	for col in df.columns[1:]:
		df[col] = df[col]/df['MakeMove']
	print('\n\nwrt MakeMove')
	print(df.mean())
	# calculate proportion again wrt create_Node
	df = pd.read_csv(outputpath)
	for col in df.columns[-5:]:
		df[col] = df[col]/df['create_Node']
	print('\n\nwrt create_Node')
	print(df.mean()[-6:])		
















