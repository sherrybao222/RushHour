''' 
BFS model self-defined ll function,
speeded version, prepared for BADS,
now working but very slow (generating data every time).
py27
'''

import MAG, time
import random, sys, copy, os, pickle
import numpy as np
import cProfile, pstats, StringIO
from operator import methodcaller
import multiprocessing as mp
from numpy import recfromcsv
from node import Node

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
	while len(n.get_children()) != 0:
		n = ArgmaxChild(n)
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
		considered_node2 = [] # new node expanded along the branch in this iteration
		while not Stop(probability=gamma) and not Determined(root):
			n, traversed = SelectNode(root)
			n2 = ExpandNode(n, theta)
			considered_node2.append(n2)
			Backpropagate(n, root)
			if n2.get_value() >= abs(np.random.normal(loc=params[mu_idx], scale=params[sigma_idx])): # terminate the algorithm if found a terminal node
				break
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

def my_ll_parallel(w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma): # parallel computing
	start_time = time.time()
	global params, num_weights, mu_idx, sigma_idx
	params = [w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma]
	num_weights = len(params)-2
	mu_idx = len(params)-2
	sigma_idx = len(params)-1
	with open("/Users/chloe/Documents/RushHour/scripts/node_list_03.pickle", "r") as fp:
		node_list = pickle.load(fp)
	with open("/Users/chloe/Documents/RushHour/scripts/expected_list_03.pickle", "r") as fp:
		expected_list = pickle.load(fp)
	# 		  [w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma]
	# trial_start = 2072 #1189 # starting row number in the raw data
	# trial_end = 2114 #1216
	# sub_data = recfromcsv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	# # construct initial node
	# dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
	# node_list = [] # list of node from data
	# expected_list = [] # list of expected human move node, str
	# cur_node = None
	# cur_carlist = None
	# for i in range(trial_start-2, trial_end-1):
	# 	# load data from datafile
	# 	row = sub_data[i]
	# 	if row['event'] == 'start':
	# 		instance = row['instance']
	# 		ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+instance+'.json'
	# 		initial_car_list, _ = MAG.json_to_car_list(ins_file)
	# 		initial_node = Node(initial_car_list)
	# 		cur_node = initial_node
	# 		cur_carlist = initial_car_list
	# 		continue
	# 	piece = row['piece']
	# 	move_to = int(row['move'])
	# 	node_list.append(cur_node) # previous board position
	# 	# create human move
	# 	cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
	# 	cur_node = Node(cur_carlist)
	# 	expected_list.append(cur_node.board_to_str())
	# node_list = np.load('/Users/chloe/Documents/RushHour/scripts/node_list.npy', allow_pickle=True)
	# expected_list = np.load('/Users/chloe/Documents/RushHour/scripts/expected_list.npy', allow_pickle=True)
	pool = mp.Pool(processes=mp.cpu_count())
	all_ibs_obj = [pool.apply_async(ibs, args=(cur, exp_str)) for cur, exp_str in zip(node_list, expected_list)]
	all_ibs_result = [r.get() for r in all_ibs_obj]
	all_ll = pool.map(harmonic_sum, [n for n in all_ibs_result])
	pool.close()
	pool.join()
	print('sampl size '+str(len(all_ll)))
	# print('all_ll: '+str(all_ll))
	ll_result = -np.sum(all_ll)
	# print('ll_result: '+str(ll_result))
	print('time '+str(time.time() - start_time))
	return ll_result

# def my_ll_sequential(w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma): # non-parallel
# 	start_time = time.time()
# 	global params, num_weights, mu_idx, sigma_idx
# 	ll = 0
# 	params = [w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma]
# 	num_weights = len(params)-2
# 	mu_idx = len(params)-2
# 	sigma_idx = len(params)-1
# 	with open("/Users/chloe/Documents/RushHour/scripts/node_list_02.pickle", "r") as fp:
# 		node_list = pickle.load(fp)
# 	with open("/Users/chloe/Documents/RushHour/scripts/expected_list_02.pickle", "r") as fp:
# 		expected_list = pickle.load(fp)
# 	# node_list = np.load('/Users/chloe/Documents/RushHour/scripts/node_list.npy', allow_pickle=True)
# 	# expected_list = np.load('/Users/chloe/Documents/RushHour/scripts/expected_list.npy', allow_pickle=True)
# 	# every human move in the trial
# 	for cur_node, exp_str in zip(node_list, expected_list):
# 		# log likelihood calculation
# 		num_simulated = ibs(cur_node, expected_board=exp_str)
# 		ll += harmonic_sum(num_simulated)
# 		# print('ibs: '+str(num_simulated))
# 	# print('params\n', params)
# 	print('ll\n', ll)
# 	print(time.time() - start_time)
# 	return -ll

def dumb_func(w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma):
	global params, num_weights, mu_idx, sigma_idx
	params = [w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma]
	num_weights = len(params)-2
	mu_idx = len(params)-2
	sigma_idx = len(params)-1
	print('I am so dumb.')
	return w0+w1+w2+w3+w4+w5+w6+w7+mu+sigma

def wrap_make_move(this_params, node):
	global params, num_weights, mu_idx, sigma_idx
	params = this_params
	num_weights = len(params)-2
	mu_idx = len(params)-2
	sigma_idx = len(params)-1
	params = this_params
	n, _, _ = MakeMove(node)
	return n

def create_data():
	trial_start = 2 # 2072 # starting row number in the raw data
	trial_end = 65 # 2114
	sub_data = recfromcsv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	# construct initial node
	node_list = [] # list of node from data
	expected_list = [] # list of expected human move node, str
	cur_node = None
	cur_carlist = None
	for i in range(trial_start-2, trial_end-1):
		# load data from datafile
		row = sub_data[i]
		if row['event'] == 'start':
			instance = row['instance']
			ins_file = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'+instance+'.json'
			initial_car_list, _ = MAG.json_to_car_list(ins_file)
			initial_node = Node(initial_car_list)
			cur_node = initial_node
			cur_carlist = initial_car_list
			continue
		piece = row['piece']
		move_to = int(row['move'])
		node_list.append(cur_node) # previous board position
		# create human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_node = Node(cur_carlist)
		expected_list.append(cur_node.board_to_str())
	# save list
	with open("/Users/chloe/Documents/RushHour/scripts/node_list_03.pickle", "w") as fp:
		pickle.dump(node_list, fp)
	with open("/Users/chloe/Documents/RushHour/scripts/expected_list_03.pickle", "w") as fp:
		pickle.dump(expected_list, fp)

# if __name__ == '__main__':
	# my_ll_parallel(-0.5391, -3.9844, 3.8281, -5.5469, -2.5781, -8.8867, -2.5469, -0.7656, 1.4844, 5.3320)
	# create_data()
	# print('---- started 03 ----')
	# with open("/Users/chloe/Documents/RushHour/scripts/node_list_03.pickle", "r") as fp:
	# 	node_list = pickle.load(fp)
	# print(type(node_list[0]))
	# print(node_list[0])
	# with open("/Users/chloe/Documents/RushHour/scripts/node_list.pickle", "r") as fp:
	# 	expected_list = pickle.load(fp)
	# print(type(expected_list[0]))
	# print(expected_list[0])






