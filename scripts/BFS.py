''' 
BFS model self-defined ll function,
speeded version, prepared for BADS in MATLAB,
python3 or py27
'''
import random, copy, pickle, os, sys, time
from operator import attrgetter
import multiprocessing as mp
import numpy as np
from numpy import recfromcsv
from json import dump, load
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
	all_moves = all_legal_moves(node.car_list, node.board)
	root_car_list = node.car_list
	for (tag, pos) in all_moves:
		new_list, _ = move_xy(root_car_list, tag, pos[0], pos[1])
		child = Node(new_list, params)
		child.parent = node
		node.children.append(child)

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
	return max(node.children, key=attrgetter('value'))
	

def MakeMove(root, params, hit=False):
	''' 
	`	returns an optimal move to make next 
		according to value function and current board position
	'''
	if hit: # for ibs, if matches human decision, return root node
		return root
	assert len(root.children) == 0
	if Lapse(params.lapse_rate): # random move
		return RandomMove(root, params)
	else:
		DropFeatures(params.feature_dropping_rate)
		while not Stop(probability=params.stopping_probability):
			leaf, leaf_is_solution = SelectNode(root)
			if leaf_is_solution:
				Backpropagate(leaf.parent, root)
				continue
			ExpandNode(leaf, params)
			Backpropagate(leaf, root)
	if root.children == []: # if did not enter while loop at all
		ExpandNode(root, params)
	return ArgmaxChild(root)



##################################### FITTING ##########################
def ibs_original(root_car_list, expected_board, params):
	''' 
		inverse binomial sampling: 
		return the number of simulations until hit target
	'''
	num_simulated = 0
	while True:
		root_node = Node(root_car_list, params)
		model_decision = MakeMove(root_node, params)
		num_simulated += 1
		if model_decision.board_to_str() == expected_board:
			return num_simulated

def harmonic_sum(n):
	''' 
		return sum of harmonic series from 1 to n-1
		when n=1, return 0
	'''
	s = 0.0
	for i in range(1, n):
		s += 1.0/i
	return s


def ibs_early_stopping(list_carlist, user_choice, inparams, pool, fun=MakeMove): # parallel computing
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
							w4=inparams[3], w5=inparams[4], w6=inparams[5], 
							w7=inparams[6], 
							stopping_probability=inparams[7],
							pruning_threshold=inparams[8],
							lapse_rate=inparams[9],
							feature_dropping_rate=0.0,
							mu=0.0, sigma=1.0)
	sys.setrecursionlimit(10000)
	start_time = time.time()
	list_rootnode = [Node(cur_root, params) for cur_root in list_carlist]
	list_answer = [Node(cl, params).board_to_str() for cl in user_choice]
	print('cpu count: '+str(mp.cpu_count()))
	# calculate early stopping LL
	hit_target = [False]*len(list_rootnode)
	count_iteration = [1]*len(list_rootnode)
	print('Sample size '+str(len(list_rootnode)))
	LL_lower = 0
	children_count = []
	for node in list_rootnode:
		children_count.append(len(all_legal_moves(node.car_list, node.board)))
	LL_lower = np.sum([np.log(1/n) for n in children_count])
	print('LL_lower '+str(LL_lower))
	print('inparams '+str(inparams))
	# start iteration
	k = 0
	LL_k = 0
	while hit_target.count(False) > 0:
		# start_time_k = time.time()
		if LL_k < LL_lower: 
			LL_k = LL_lower
			print('******************* Exceeds LL_lower, break')
			break
		LL_k = 0
		k += 1
		print('Iteration K='+str(k))	
		list_rootnode = [Node(cur_root, params) for cur_root in list_carlist]
		model_decision = [pool.apply_async(fun, args=(cur_root, params, hit)).get() for cur_root, hit in zip(list_rootnode, hit_target)]
		hit_target = [a or b for a,b in zip(hit_target, [decision.board_to_str()==answer for decision, answer in zip(model_decision, list_answer)])]
		for i in range(len(count_iteration)):
			if not hit_target[i]:
				count_iteration[i] += 1
		# new_hit = [False]*len(list_rootnode)
		#  new_hit[:min(k*5, len(list_rootnode)-1)] = [True]*min(k*5, len(list_rootnode)-1)
		# hit_target = [a or b for a,b in zip(hit_target, new_hit)]
		for i in range(len(count_iteration)):
			if hit_target[i]:
				LL_k += harmonic_sum(count_iteration[i])
		LL_k = (-1.0)*LL_k - (hit_target.count(False))*harmonic_sum(k)
		print('\thit_target '+str(hit_target.count(True)))
		print('\tKth LL_k '+str(LL_k))
		# print('\tIBS kth iteration lapse '+str(time.time() - start_time_k))	
	print('IBS total time lapse '+str(time.time() - start_time))
	print('now time: '+str(datetime.now()))
	print('Final LL_k: '+str(LL_k))
	return LL_k


def ibs_early_stopping_sequential(list_carlist, user_choice, inparams): # parallel computing
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
							w4=inparams[3], w5=inparams[4], w6=inparams[5], 
							w7=inparams[6], 
							stopping_probability=inparams[7],
							pruning_threshold=inparams[8],
							lapse_rate=inparams[9])
	sys.setrecursionlimit(10000)
	start_time = time.time()
	list_rootnode = [Node(cur_root, params) for cur_root in list_carlist]
	list_answer = [Node(cl, params).board_to_str() for cl in user_choice]
	print('cpu count: '+str(mp.cpu_count()))
	# calculate early stopping LL
	hit_target = [False]*len(list_rootnode)
	count_iteration = [0]*len(list_rootnode)
	print('Sample size '+str(len(list_rootnode)))
	LL_lower = 0
	children_count = []
	for node in list_rootnode:
		children_count.append(len(all_legal_moves(node.car_list, node.board)))
	LL_lower = np.mean([np.log(1.0/n) for n in children_count])
	print('LL_lower '+str(LL_lower))
	print('Params sp '+str(params.stopping_probability))
	count_iteration = [x+1 for x in count_iteration]
	# start iteration
	k = 0
	LL_k = 0
	while hit_target.count(False) > 0:
		start_time_k = time.time()
		if LL_k < LL_lower: 
			LL_k = LL_lower
			print('Exceeds LL_lower, break')
			break
		LL_k = 0
		k += 1
		print('Iteration K='+str(k))	
		list_rootnode = [Node(cur_root, params) for cur_root in list_carlist]
		model_decision = []
		for cur_root, hit in zip(list_rootnode, hit_target):
			model_decision.append(MakeMove(cur_root, params, hit))
		print('post makemove')
		for i in range(len(count_iteration)):
			if not hit_target[i]:
				count_iteration[i] += 1
		hit_target = [a or b for a,b in zip(hit_target, [decision.board_to_str()==answer for decision, answer in zip(model_decision, list_answer)])]
		for i in range(len(count_iteration)):
			if hit_target[i]:
				LL_k += harmonic_sum(count_iteration[i])
		# print('\thit_target.count(False): '+str(hit_target.count(False)))
		LL_k = (-1.0/len(hit_target))*LL_k - (hit_target.count(False)/len(hit_target))*harmonic_sum(k)
		print('\thit_target '+str(hit_target.count(True)))
		print('\tKth LL_k '+str(LL_k))
		print('\tIBS kth iteration lapse '+str(time.time() - start_time_k))	
		# if k >= 10:
		# 	print('exceed 10 iterations, break')
		# 	break
	print('IBS total time lapse '+str(time.time() - start_time))
	print('now time: '+str(datetime.now()))
	print('Final LL_k: '+str(LL_k))
	return LL_k


def my_ll_parallel(params): # parallel computing
	sys.setrecursionlimit(10000)
	start_time = time.time()
	list_carlist, user_choice = load_data()
	pool = mp.Pool(processes=mp.cpu_count())
	print('cpu count: '+str(mp.cpu_count()))
	# calculate early stopping LL
	hit_target = [False]*len(list_carlist)
	count_iteration = [0]*len(list_carlist)
	print('Sample size '+str(len(list_carlist)))
	list_answer = [Node(cl, params).board_to_str() for cl in user_choice] # str
	print('\tParams sp '+str(params.stopping_probability))
	all_ibs = [pool.apply_async(ibs, args=(cur, exp_str, params)).get() for cur, exp_str in zip(list_carlist, list_answer)]
	pool.close()
	pool.join()
	# print('\tNum: '+str(all_ibs))
	print('\tAvg Num: '+str(np.mean(all_ibs)))
	all_ll = []
	for n in all_ibs:
		all_ll.append(harmonic_sum_Luigi(n))
	ll_result = -np.sum(all_ll)
	print('\ttime lapse '+str(time.time() - start_time))
	print('\tnow time: '+str(datetime.now()))
	print('\tFinal LL: '+str(ll_result))
	return ll_result


def my_ll_sequential(list_carlist, user_choice, inparams): # parallel computing
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
							w4=inparams[3], w5=inparams[4], w6=inparams[5], 
							w7=inparams[6], 
							stopping_probability=inparams[7],
							pruning_threshold=inparams[8],
							lapse_rate=inparams[9],
							feature_dropping_rate=0.0,
							mu=0.0, sigma=1.0)
	sys.setrecursionlimit(10000)
	# start_time = time.time()
	# calculate early stopping LL
	hit_target = [False]*len(list_carlist)
	count_iteration = [0]*len(list_carlist)
	# print('Sample size '+str(len(list_carlist)))
	list_answer = [Node(cl, params).board_to_str() for cl in user_choice] # str
	print(inparams)
	all_ibs = []
	# t = []
	for curlist, answer in zip(list_carlist, list_answer):
		# t1 = time.time()
		all_ibs.append(ibs(curlist, answer, params))
		# print('\tll for one trial: '+str(time.time()-t1))
		# t.append(time.time()-t1)
	# print('\tAvg time per trial '+str(np.mean(t)))
	print('\tAvg Num: '+str(np.mean(all_ibs)))
	all_ll = []
	for n in all_ibs:
		all_ll.append(harmonic_sum_Luigi(n))
	ll_result = -np.sum(all_ll)
	# print('\ttime lapse '+str(time.time() - start_time))
	print('\tnow time: '+str(datetime.now()))
	print('\tFinal LL: '+str(ll_result))
	return ll_result

if __name__ == "__main__":

	all_positions = pickle.load(open('/Users/chloe/Desktop/carlists/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9_positions.pickle', 'rb'))[:300]
	all_decisions = pickle.load(open('/Users/chloe/Desktop/carlists/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9_decisions.pickle', 'rb'))[:300]

	kfold = 3

	guesses = []
	for k in range(kfold):
		guesses.append(np.array([random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),random.random(),
						random.random(), random.random()*10, random.random()], np.float))

	kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
	positions_train = []
	positions_test = []
	decisions_train = []
	decisions_test = []
	for train_index, test_index in kf.split(all_positions):
		positions_train.append([all_positions[i] for i in train_index])
		positions_test.append([all_positions[i] for i in test_index])
		decisions_train.append([all_decisions[i] for i in train_index])
		decisions_test.append([all_decisions[i] for i in test_index])

	results = []
	for k in range(kfold):
		pool = mp.Pool(processes=mp.cpu_count())
		# def MLERegression(params):
		# 	return ibs_early_stopping(positions_train[k], decisions_train[k], params, pool)
		# result = minimize(MLERegression, guesses[k], method = 'Nelder-Mead',
		# 					options={'disp': True})

		bounds = [(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,2),(0,1),(0,20),(0,1)]
		u = np.array((0,0,0,0,0,0,0,0,0,0), np.float) # lower bound
		v = np.array((2,2,2,2,2,2,2,1,20,1), np.float) # upper bound
		def func(params):
			return ibs_early_stopping(positions_train[k], decisions_train[k], params, pool)
		# result = optimize.dual_annealing(func, bounds)
		result = differential_evolution(func, bounds)

		print(result)
		results.append(result)
		pool.join()
		pool.close()



# def MakeMove_plot(root, params):
# 	''' 
# 		MakeMove function with movie plotting arguments,
# 		work together with script plot_movie.py
# 	`	returns an optimal move to make next 
# 		according to value function, 
# 		current state given
# 	'''
# 	assert len(root.children) == 0
# 	if Lapse(params.lapse_rate):
# 		random_choice = RandomMove(root, params)
# 		plot_board_and_tree(root, board_node=root, text='Random Move', text2='Initial Board')
# 		plot_board_and_tree(root, board_node=random_choice, highlighted_node=random_choice, text='Random Move', text2='Move 1')
# 		return random_choice
# 	else:
# 		DropFeatures(params.feature_dropping_rate)
# 		hash_most_promising_node = None
# 		while not Stop(probability=params.stopping_probability):
# 		# for i in range(4):
# 			leaf, leaf_is_solution = SelectNode(root)
# 			if leaf == hash_most_promising_node: # expand along last principal variation
# 				plot_board_and_tree(root, board_node=leaf, highlighted_node=leaf, highlighted_edges=leaf.find_path_to_root(), text='Select', text2='Move '+str(move_num))
# 				move_num += 1
# 			else: 
# 				if hash_most_promising_node:
# 					plot_blank(text='New Principal Variation')
# 				plot_board_and_tree(root, board_node=root, text='Initial State', text2='Initial Board')
# 				move_num = 0
# 				for n in leaf.find_path_to_root():
# 					if move_num == 0:
# 						plot_board_and_tree(root, board_node=n, highlighted_node=n, highlighted_edges=n.find_path_to_root(), text='Select', text2='Initial Board')
# 					else:
# 						plot_board_and_tree(root, board_node=n, highlighted_node=n, highlighted_edges=n.find_path_to_root(), text='Select', text2='Move '+str(move_num))
# 					move_num += 1
# 			if leaf_is_solution:
# 				print('solution found, backprop and break')
# 				plot_board_and_tree(root, board_node=leaf, highlighted_node=leaf, highlighted_edges=leaf.find_path_to_root(), text='Solution Found', text2='Solved')
# 				Backpropagate(leaf.parent, root)
# 				for n in leaf.find_path_to_root()[::-1][1:]:
# 					plot_board_and_tree(root, board_node=leaf, highlighted_node=leaf, updated_node=[n], highlighted_edges=leaf.find_path_to_root(), text='Backpropagate', text2='Solved')
# 				break
# 			ExpandNode(leaf, params)
# 			hash_most_promising_node = ArgmaxChild(leaf)
# 			if move_num-1 != 0:
# 				plot_board_and_tree(root, board_node=leaf, highlighted_node=leaf, highlighted_edges=leaf.find_path_to_root(), text='Expand', text2='Move '+str(move_num-1))
# 			if move_num-1 == 0:
# 				plot_board_and_tree(root, board_node=leaf, highlighted_node=leaf, highlighted_edges=leaf.find_path_to_root(), text='Expand', text2='Initial Board')
# 			plot_board_and_tree(root, board_node=leaf, board_to_node=ArgmaxChild(leaf), highlighted_node=ArgmaxChild(leaf), highlighted_edges=ArgmaxChild(leaf).find_path_to_root(), text='Most Promising Node', text2='Move '+str(move_num))
# 			plot_board_and_tree(root, board_node=ArgmaxChild(leaf), highlighted_node=ArgmaxChild(leaf), highlighted_edges=ArgmaxChild(leaf).find_path_to_root(), text='Most Promising Node', text2='Move '+str(move_num))
# 			Backpropagate(leaf, root)
# 			for n in leaf.find_path_to_root()[::-1]:
# 				plot_board_and_tree(root, board_node=ArgmaxChild(leaf), highlighted_node=ArgmaxChild(leaf), updated_node=[n], highlighted_edges=ArgmaxChild(leaf).find_path_to_root(), text='Backpropagate', text2='Move '+str(move_num))
# 			print('\titeration')
# 		print('Stop')
# 	if root.children == []:
# 		ExpandNode(root, params)
# 	plot_blank(text='Best-First Search terminated.\n        Predicted move:')
# 	plot_board_and_tree(root, board_node=root, board_to_node=ArgmaxChild(root), decision_node=root, text='Decision', text2='Initial Board')
# 	plot_board_and_tree(root, board_node=ArgmaxChild(root), decision_node=ArgmaxChild(root), text='Decision', text2='Move 1')
# 	make_movie()
# 	return ArgmaxChild(root)



