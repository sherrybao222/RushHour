''' 
BFS model, 
speeded version, ready for parameter fitting
py27
'''

import MAG, rushhour
import random, sys, copy, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Digraph
from scipy.optimize import minimize
import cProfile, pstats, StringIO
from operator import methodcaller
import multiprocessing as mp


class Node:
	def __init__(self, cl):
		self.__car_list = cl
		self.__children = []
		self.__value = None
		self.__red = None
		self.__board = None #str
	def add_child(self, n):
		n.set_parent(self)
		self.__children.append(n)
	def set_parent(self, p):
		self.__parent = p
	def set_value(self, v):
		self.__value = v
	def get_carlist(self):
		return self.__car_list
	def get_red(self):
		if self.__red == None:
			for car in self.__car_list:
				if car.tag == 'r':
					self.__red = car
		return self.__red
	def get_board(self):
		tmp_b, _ = MAG.construct_board(self.__car_list)
		return tmp_b
	def get_value(self):
		if self.__value == None:
			self.__value = Value1(self.__car_list, self.get_red())
		return self.__value
	def get_child(self, ind):
		return self.__children[ind]
	def get_children(self):
		return self.__children
	def find_child(self, c):
		for i in range(len(self.__children)):
			if self.__children[i] == c:
				return i
		return None
	def find_child_by_str(self, bstr):
		for i in range(len(self.__children)):
			if self.__children[i].board_to_str() == bstr:
				return i
		return None
	def get_parent(self):
		return self.__parent
	def remove_child(self, c):
		for i in range(len(self.__children)):
			if self.__children[i] == c:
				c.parent = None
				self.__children.pop(i)
				return	
	def board_to_str(self):
		if self.__board == None:
			tmp_board, tmp_red = MAG.construct_board(self.__car_list)
			out_str = ''
			for i in range(tmp_board.height):
				for j in range(tmp_board.width):
					cur_car = tmp_board.board_dict[(j, i)]
					if cur_car == None:
						out_str += '.'
						if i == 2 and j == 5:
							out_str += '>'
						continue
					if cur_car.tag == 'r':
						out_str += 'R'
					else:
						out_str += cur_car.tag
					if i == 2 and j == 5:
						out_str += '>'
				out_str += '\n'
			self.__board = out_str
		return self.__board

def Value1(car_list2, red):
	'''
	value = w0 * num_cars{MAG-level 0/RED}
		+ w1 * num_cars{MAG-level 1} 
		+ w2 * num_cars{MAG-level 2}  
		+ w3 * num_cars{MAG-level 3} 
		+ w4 * num_cars{MAG-level 4} 
		+ w5 * num_cars{MAG-level 5} 
		+ w6 * num_cars{MAG-level 6}
		+ w7 * num_cars{MAG-level 7}  
		+ noise
	'''
	noise = np.random.normal(loc=params[mu_idx], scale=params[sigma_idx])
	# initialize MAG
	my_board2, my_red2 = MAG.construct_board(car_list2)
	new_car_list2 = MAG.construct_mag(my_board2, my_red2)
	# each following level
	new_car_list2 = MAG.assign_level(new_car_list2, my_red2)
	value = np.sum(np.array(MAG.get_num_cars_from_levels(new_car_list2, num_weights)) * np.array(params[:num_weights]))
	return value+noise

def DropFeatures(delta):
	pass

def Lapse(probability=0.05): 
	''' return true with a probability '''
	return random.random() < probability

def Stop(probability=0.05): 
	''' return true with a probability '''
	return random.random() < probability

def Determined(root_node): 
	''' return true if win, false otherwise '''
	return MAG.check_win(root_node.get_board(), root_node.get_red())

def RandomMove(node):
	''' make a random move and return the resulted node '''
	return random.choice(node.get_children())
	
def InitializeChildren(root_node):
	''' initialize the list of children (using all possible moves) '''
	if len(root_node.get_children()) == 0:
		all_moves = MAG.all_legal_moves(root_node.get_carlist(), root_node.get_red(), root_node.get_board())
		root_car_list = root_node.get_carlist()
		for i, (tag, pos) in enumerate(all_moves):
			new_list, _ = MAG.move2(root_car_list, tag, pos[0], pos[1])
			dummy_child = Node(new_list)
			root_node.add_child(dummy_child)

def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	if plot_flag:
		traversed = []
	while len(n.get_children()) != 0:
		n = ArgmaxChild(n)
		if plot_flag:
			traversed.append(n)
	if plot_flag:
		return n, traversed
	return n, []
 
def ExpandNode(node, threshold):
	''' create all possible nodes under input node, 
	cut the ones below threshold '''
	if len(node.get_children()) == 0:
		InitializeChildren(node)
	Vmaxchild = ArgmaxChild(node)
	Vmax = Vmaxchild.get_value()
	for child in node.get_children():
		if abs(child.get_value() - Vmax) > threshold:
			node.remove_child(child)
	return Vmaxchild

def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	this_node.set_value(ArgmaxChild(this_node).get_value())
	if this_node != root_node:
		Backpropagate(this_node.get_parent(), root_node)

def ArgmaxChild(root_node): 
	''' return the child with max value '''
	return max(root_node.get_children(), key=methodcaller('get_value'))
	
def MakeMove(state, delta=0, gamma=0.05, theta=10):
	''' returns an optimal move to make next, given current state '''
	root = state # state is already a node
	InitializeChildren(root)
	if Lapse():
		return RandomMove(root), [], []
	else:
		DropFeatures(delta)
		if plot_flag:
			considered_node = [] # nodes already traversed along the branch in this ieration
		considered_node2 = [] # new node expanded along the branch in this iteration
		while not Stop(probability=gamma) and not Determined(root):
			n, traversed = SelectNode(root)
			if plot_flag:
				considered_node.append(traversed)
			n2 = ExpandNode(n, theta)
			considered_node2.append(n2)
			Backpropagate(n, root)
			if n2.get_value() >= abs(np.random.normal(loc=params[mu_idx], scale=params[sigma_idx])): # terminate the algorithm if found a terminal node
				break
	if plot_flag:
		return ArgmaxChild(root), considered_node, considered_node2
	return ArgmaxChild(root), [], considered_node2

def ibs(root_node, expected_board=''):
	''' 
		inverse binomial sampling: 
		return the number of simulations until hit target
	'''
	InitializeChildren(root_node)
	num_simulated = 0
	hit = False
	while not hit:
		new_node, _, _ = MakeMove(root_node)
		num_simulated += 1
		if new_node.board_to_str() == expected_board:
			hit = True
	return num_simulated

def harmonic_sum(n):
	''' return sum of harmonic series from 1 to k '''
	i = 1
	s = 0.0
	for i in range(1, n+1):
		s += 1.0/i
	return s

def my_ll_parallel(params): # parallel computing
	ll_one_sim = []
	num_sim[0] += 1
	print('-------- parallel simulation '+str(num_sim[0])+' ---------')
	all_ibs_obj = [pool.apply_async(ibs, args=(cur, exp_str)) for cur, exp_str in zip(node_list, expected_list)]
	# for cur, exp_str in zip(node_list, expected_list):
	# 	pool.apply_async(ibs, args=(cur, exp_str), callback=collect_result)
	all_ibs_result = [r.get() for r in all_ibs_obj]
	print('all_ibs_result', all_ibs_result)
	all_ll = pool.map(harmonic_sum, [n for n in all_ibs_result])
	print('params', params)
	return -np.sum(all_ll)

def collect_result(r):
	# call back function for parallel result collection
	global ll_one_sim
	ll_one_sim.append(r)


if __name__ == '__main__':
	# 		  [w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma]
	num_sim = [0]
	params = [0, -8, -10, -5, -3, -1, -1, -1, 0, 4]
	num_weights = len(params)-2
	mu_idx = len(params)-2
	sigma_idx = len(params)-1
	trial_start = 2 # starting row number in the raw data
	trial_end = 20
	plot_flag = False
	sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')

	# construct initial node
	dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
	node_list = [] # list of node from data
	expected_list = [] # list of expected human move node, str
	cur_node = None
	cur_carlist = None
	ll_one_sim = [] # ll result from one simulation
	for i in range(trial_start-2, trial_end-2):
		# load data from datafile
		row = sub_data.loc[i, :]
		if row['event'] == 'start':
			# print row
			instance = row['instance']
			ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+instance+'.json'
			initial_car_list, _ = MAG.json_to_car_list(ins_file)
			initial_node = Node(initial_car_list)
			cur_node = initial_node
			cur_carlist = initial_car_list
			continue
		piece = row['piece']
		move_to = row['move']
		node_list.append(cur_node) # previous board position
		# create human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_node = Node(cur_carlist)
		expected_list.append(cur_node.board_to_str())
	print('====================== started =====================')


	# pr = cProfile.Profile()
	# pr.enable()
		
	# parallel processing	
	pool = mp.Pool(processes=mp.cpu_count())

	# fit
	# results = minimize(my_ll_parallel, params, 
	# 		method='Nelder-Mead', options={'disp': True})	
	results = minimize(my_ll_parallel, params, 
			method='BFGS', options={'eps': 0.1})	
	print(results)
	pool.close()
	pool.join()

	# pr.disable()
	# s = StringIO.StringIO()
	# sortby = 'cumulative'
	# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	# ps.print_stats()
	# print s.getvalue()















def estimate_prob(root_node, expected_board='', iteration=100):
	''' Estimate the probability of next possible moves given the root node '''
	# InitializeChildren(root_node)
	first_iteration = True
	frequency = None
	sol_idx = None
	
	for i in range(iteration):
		new_node, _, _ = MakeMove(root_node)
		if first_iteration:
			frequency = [0] * len(root_node.get_children())
			first_iteration = False
		child_idx = root_node.find_child(new_node)
		frequency[child_idx] += 1
	# turn frequency into probability
	frequency = np.array(frequency, dtype=np.float32)/iteration 
	for i in range(len(root_node.get_children())):
		if root_node.get_child(i).board_to_str() == expected_board:
			sol_idx = i
	return root_node.get_children(), frequency, sol_idx, [], []



def my_ll_sequential(params): # non-parallel
	# pr = cProfile.Profile()
	# pr.enable()
	print('---------------- sequential simulation ----------------')
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1
	ll = 0
	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		# load data from datafile
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']
		# create human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		# log likelihood calculation
		num_simulated = ibs(cur_node, 
				expected_board=Node(cur_carlist).board_to_str())
		ll += harmonic_sum(num_simulated)
		print('ibs: '+str(num_simulated))
		# make human move node
		cur_node = Node(cur_carlist)
		move_num += 1
	# pr.disable()
	# s = StringIO.StringIO()
	# sortby = 'cumulative'
	# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	# ps.print_stats()
	# print s.getvalue()
	print params
	return -ll

