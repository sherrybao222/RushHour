''' 
BFS model self-defined ll function,
speeded version, prepared for BADS in MATLAB,
now working but very slow (generating data every time).
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
# from plot_movie import *
import pandas as pd
# from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize
import scipy.stats as stats


######################################## MAG CONTENT ##############

class Car:
	def __init__(self, s, l, t, o, p):
		self.start = s # [hor, ver]
		self.length = l # int
		self.tag = t # str
		self.orientation = o # str
		self.puzzle_tag = p # str
		if self.orientation == 'horizontal':
			self.end = [self.start[0] + self.length - 1, self.start[1]]
		elif self.orientation == 'vertical':
			self.end = [self.start[0], self.start[1] + self.length - 1]
		self.edge_to = []
		self.level = []
		self.visited = False

class Board:
	def __init__(self, car_list):
		self.height = 6
		self.width = 6
		self.board_dict = {}
		for i in range(0, self.height):
			for j in range(0, self.width):
				self.board_dict[(j, i)] = None
		for car in car_list:
			if car.tag == 'r':
				self.red = car
			occupied_space = []
			if car.orientation == 'horizontal':
				for i in range(car.length):
					occupied_space.append((car.start[0] + i, car.start[1]))
			elif car.orientation == 'vertical':
				for i in range(car.length):
					occupied_space.append((car.start[0], car.start[1] + i))
			for xy in occupied_space:
				self.board_dict[xy] = car

def json_to_car_list(filename):
	with open(filename,'r') as data_file:
		car_list = []
		data = load(data_file)
		for c in data['cars']:
			cur_car = Car(s = [int(c['position'])%6, int(c['position']/6)], \
				l = int(c['length']), t = str(c['id']), o = c['orientation'], p = data['id'])
			car_list.append(cur_car)
	return car_list

def carlist_to_board_str(carlist):
	board = Board(carlist)
	out_str = ''
	for i in range(board.height):
		for j in range(board.width):
			cur_car = board.board_dict[(j, i)]
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
	return out_str

def move(car_list, car_tag, to_position): 
	'''
		make a move and return the new car list, 
		single position label
	'''
	new_list2 = []
	for cur_car in car_list:
		if cur_car.tag == car_tag:
			new_car = Car(s = [int(to_position)%6, int(to_position/6)],\
				l = int(cur_car.length), t = car_tag, \
				o = cur_car.orientation, p = cur_car.puzzle_tag)
		else:
			new_car = Car(s = [int(cur_car.start[0]), int(cur_car.start[1])], \
							l = int(cur_car.length), t = cur_car.tag,\
							o = cur_car.orientation, p = cur_car.puzzle_tag)
		new_list2.append(new_car)
		if new_car.tag == 'r':
			red = new_list2[-1]
	return new_list2, red

def move_xy(car_list, car_tag, to_position1, to_position2): 
	'''
		make a move and return the new car list, x and y
	'''
	new_list2 = []
	for cur_car in car_list:
		if cur_car.tag == car_tag:
			new_car = Car(s = [int(to_position1), int(to_position2)],\
				l = int(cur_car.length), t = car_tag, \
				o = cur_car.orientation, p = cur_car.puzzle_tag)
		else:
			new_car = Car(s = [int(cur_car.start[0]), int(cur_car.start[1])], \
							l = int(cur_car.length), t = cur_car.tag,\
							o = cur_car.orientation, p = cur_car.puzzle_tag)
		new_list2.append(new_car)
		if new_car.tag == 'r':
			red = new_list2[-1]
	return new_list2, red

def all_legal_moves(car_list, board):
	moves = []
	for i in range(len(car_list)):
		cur_car = car_list[i]
		if cur_car.orientation == 'horizontal':
			cur_position1 = cur_car.start[0] - 1 # search left
			cur_position2 = cur_car.start[1]
			while(cur_position1 >= 0 and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1, cur_position2]))
				cur_position1 -= 1
			cur_position1 = cur_car.end[0] + 1 # search right
			while(cur_position1 < board.width and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1-cur_car.length+1, cur_position2]))
				cur_position1 += 1
		if cur_car.orientation == 'vertical':
			cur_position1 = cur_car.start[0]
			cur_position2 = cur_car.start[1] - 1 # search up
			while(cur_position2 >= 0 and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1, cur_position2]))
				cur_position2 -= 1
			cur_position2 = cur_car.end[1] + 1 # searc down
			while(cur_position2 < board.height and board.board_dict[(cur_position1, cur_position2)] == None):
				moves.append((cur_car.tag, [cur_position1, cur_position2-cur_car.length+1]))
				cur_position2 += 1
	return moves
			
def is_solved(board, red): 
	'''
		return true if current board state can win
		no car in front of red
	'''
	cur_position = red.end[0] + 1 
	while(cur_position < board.width): # search right of red car
		if board.board_dict[(cur_position, red.start[1])] is not None:
			return False
		cur_position += 1
	return True

def assign_level(car_list): 
	''' 
	assign level to each car
	'''
	for car in car_list: # clean levels of each car 
		car.level = []
		if car.tag == 'r': # find red
			red = car
	queue = []
	visited = []
	red.level.append(0)
	queue.append(red)
	visited.append(red)
	while queue: # bfs, assign levels
		cur_car = queue.pop(0)
		for child in cur_car.edge_to:
			child.level.append(cur_car.level[-1] + 1)
			if child not in visited:
				queue.append(child)
				visited.append(child)

def get_num_cars_from_levels(car_list, highest_level):
	''' 
		return the number of cars at each level 
		highest_level >= any possible min(cur_car.level)
	'''
	list_toreturn = [0] * (highest_level+1)
	for cur_car in car_list:
		if cur_car.level != []:
			list_toreturn[min(cur_car.level)] += 1
	return list_toreturn

def construct_mag(board, red):
	'''
		assign graph edges and neighbors, return a new car list
	'''
	queue = []
	i = board.width - 1
	for i in range(red.end[0], board.width): # obstacles in front of red, include red
		cur_car = board.board_dict[(i, red.end[1])]
		if cur_car is not None: #exists on board
			queue.append(cur_car)
			if cur_car.tag != 'r':
				red.edge_to.append(cur_car)
				cur_car.visited = True
	red.visited = True
	while len(queue) != 0: # obstacle blockers
		cur_car	= queue.pop()
		if cur_car.tag == 'r':
			continue
		if cur_car.orientation == 'vertical': # vertical
			if cur_car.start[1] > 0:
				j = cur_car.start[1] - 1
				while j >= 0: # upper
					meet_car =  board.board_dict[(cur_car.start[0], j)]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					j -= 1
			if cur_car.end[1] < board.height - 1:
				k = cur_car.end[1] + 1
				while k <= board.height - 1: # lower
					meet_car =  board.board_dict[(cur_car.start[0], k)]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					k += 1
		elif cur_car.orientation == 'horizontal': # or horizontal
			if cur_car.start[0] > 0:
				j = cur_car.start[0] - 1
				while j >= 0: # left
					meet_car =  board.board_dict[(j, cur_car.start[1])]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					j -= 1
			if cur_car.end[0] < board.width - 1:
				k = cur_car.end[0] + 1
				while k <= board.width - 1: # right
					meet_car =  board.board_dict[(k, cur_car.start[1])]
					if meet_car is not None:
						if meet_car not in cur_car.edge_to:
							cur_car.edge_to.append(meet_car)
							if not meet_car.visited:
								queue.append(meet_car)
								meet_car.visited = True
					k += 1
		cur_car.visited = True # mark
	# clean all visited flags
	for i in range(0, board.height):
		for j in range(0, board.width):
			cur_car = board.board_dict[(i, j)]
			if cur_car is not None:
				cur_car.visited = False
	

######################################## BFS MODEL ##############

class Node:
	def __init__(self, cl, params):
		self.car_list = cl # list of Car
		self.parent = None # Node
		self.children = [] # list of Node
		self.board = Board(self.car_list) # str
		for car in self.car_list: # Car
				if car.tag == 'r':
					self.red = car
		construct_mag(self.board, self.red)
		assign_level(self.car_list)
		self.value = self.heuristic_value_function(params) # float
	def __members(self):
		return (self.car_list, self.children, self.value, self.red, self.board)
	def __eq__(self, other):
		if type(other) is type(self):
			return self.__members() == other.__members()
		else:
			return False
	def __hash__(self):
		return hash(self.__members())
	def find_child(self, c):
		for o in self.children:
			if o == c:
				return o
		return None
	def find_child_by_str(self, bstr):
		for c in self.children:
			if c.board_to_str() == bstr:
				return c
		return None
	def remove_child(self, c):
		for i in range(len(self.children)):
			if self.children[i] == c:
				c.parent = None
				self.children.pop(i)
				return	
	def board_to_str(self):
		out_str = ''
		for i in range(self.board.height):
			for j in range(self.board.width):
				cur_car = self.board.board_dict[(j, i)]
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
		return out_str
	def heuristic_value_function(self, params):
		'''
		value = w0 * num_cars{MAG-level RED}
			+ w1 * num_cars{MAG-level 1} 
			+ w2 * num_cars{MAG-level 2}  
			+ w3 * num_cars{MAG-level 3} 
			+ w4 * num_cars{MAG-level 4} 
			+ w5 * num_cars{MAG-level 5} 
			+ w6 * num_cars{MAG-level 6}
			+ w7 * num_cars{MAG-level 7}  
			+ noise
		'''
		value = np.sum(np.array(get_num_cars_from_levels(self.car_list, params.num_weights-1), dtype=np.int64) 
				* np.array(params.weights, dtype=np.float64), dtype=np.float64)
		noise = np.random.normal(loc=params.mu, scale=params.sigma)
		return -(value+noise)
	def find_path_to_root(self): 
		''' return path of Node from self to root '''
		trajectory = []
		cur = self
		while cur != None:
			trajectory.insert(0, cur)
			cur = cur.parent
		return trajectory
	def board_matrix(self):
		''' convert to an int matrix of board configuration '''
		matrix = np.zeros((6,6), dtype=int)
		line_idx = 0
		for line in self.board_to_str().split('\n'):
			char_idx = 0
			for char in line:
				if char == '>':
					continue
				elif char == '.':
					matrix[line_idx][char_idx] = -1
				elif char == 'R':
					matrix[line_idx][char_idx] = 0
				else:
					matrix[line_idx][char_idx] = int(char)+1
				char_idx += 1
			line_idx += 1
		matrix = np.ma.masked_where(matrix==-1, matrix)
		return matrix
	def find_changed_car(self, to_node):
		for tocar in to_node.car_list:
			for fromcar in self.car_list:
				if fromcar.tag == tocar.tag and fromcar.start != tocar.start:
					return fromcar, tocar

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
	# print('Random Move')
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
	Vmax = Vmaxchild.value
	for child in node.children:
		if abs(child.value - Vmax) > params.pruning_threshold:
			node.remove_child(child)

def Backpropagate(this_node, root_node, value):
	''' update value back until root node '''
	this_node.value = value
	if this_node != root_node:
		Backpropagate(this_node.parent, root_node, value)

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
	if hit:
		return root
	# start_time = time.time()
	assert len(root.children) == 0
	if Lapse(params.lapse_rate):
		# print('Random move made')
		# print('MakeMove Time lapse: '+format(time.time()-start_time, '.6f'))
		return RandomMove(root, params)
	else:
		DropFeatures(params.feature_dropping_rate)
		while not Stop(probability=params.stopping_probability):
			leaf, leaf_is_solution = SelectNode(root)
			if leaf_is_solution:
				Backpropagate(leaf.parent, root, leaf.value)
				break
			ExpandNode(leaf, params)
			Backpropagate(leaf, root, ArgmaxChild(leaf).value)
			# print('\tnew simulation')
		# print('Simulation terminated')
	if root.children == []:
		ExpandNode(root, params)
	# print('\t\t MakeMove time lapse: '+str(time.time()-start_time))
	return ArgmaxChild(root)

##################################### FITTING ##########################
def ibs(root_car_list, expected_board, params):
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
		elif num_simulated > 20:
			return num_simulated

def harmonic_sum_Luigi(n):
	''' 
		return sum of harmonic series from 1 to n-1
		when n=1, return 0
	'''
	s = 0.0
	for i in range(1, n):
		s += 1.0/i
	return s

def harmonic_sum(n):
	''' 
		return sum of harmonic series from 1 to n
		when n=1, return 1
	'''
	s = 1.0
	for i in range(1, n):
		s += 1.0/i
	return s

def ibs_early_stopping(list_carlist, user_choice, inparams, pool): # parallel computing
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
	count_iteration = [0]*len(list_rootnode)
	print('Sample size '+str(len(list_rootnode)))
	LL_lower = 0
	children_count = []
	for node in list_rootnode:
		children_count.append(len(all_legal_moves(node.car_list, node.board)))
	LL_lower = len(list_rootnode)*np.log(1.0/np.mean([n for n in children_count]))
	print('LL_lower '+str(LL_lower))
	print('inparams '+str(inparams))
	count_iteration = [x+1 for x in count_iteration]
	# start iteration
	k = 0
	LL_k = 0
	while hit_target.count(False) > 0:
		start_time_k = time.time()
		if LL_k < LL_lower: 
			LL_k = LL_lower
			print('******************* Exceeds LL_lower, break')
			break
		LL_k = 0
		k += 1
		print('Iteration K='+str(k))	
		list_rootnode = [Node(cur_root, params) for cur_root in list_carlist]
		model_decision = [pool.apply_async(MakeMove, args=(cur_root, params, hit)).get() for cur_root, hit in zip(list_rootnode, hit_target)]
		for i in range(len(count_iteration)):
			if not hit_target[i]:
				count_iteration[i] += 1
		hit_target = [a or b for a,b in zip(hit_target, [decision.board_to_str()==answer for decision, answer in zip(model_decision, list_answer)])]
		# new_hit = [False]*len(list_rootnode)
		# new_hit[:min(k*200, len(list_rootnode)-1)] = [True]*min(k*200, len(list_rootnode)-1)
		# hit_target = [a or b for a,b in zip(hit_target, new_hit)]
		for i in range(len(count_iteration)):
			if hit_target[i]:
				LL_k += harmonic_sum(count_iteration[i])
		LL_k = (-1.0)*LL_k - (hit_target.count(False))*harmonic_sum(k)
		print('\thit_target '+str(hit_target.count(True)))
		print('\tKth LL_k '+str(LL_k))
		print('\tIBS kth iteration lapse '+str(time.time() - start_time_k))	
	print('IBS total time lapse '+str(time.time() - start_time))
	print('now time: '+str(datetime.now()))
	print('Final LL_k: '+str(LL_k))
	return LL_k



def fake_ibs_early_stopping(list_carlist, user_choice, inparams, pool): # parallel computing
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
		model_decision = [pool.apply_async(MakeMove, args=(cur_root, params, hit)).get() for cur_root, hit in zip(list_rootnode, hit_target)]
		# print('post makemove')
		for i in range(len(count_iteration)):
			if not hit_target[i]:
				count_iteration[i] += 1
		hit_target = [a or b for a,b in zip(hit_target, [decision.board_to_str()==answer for decision, answer in zip(model_decision, list_answer)])]
		hit_target = []
		for i in range(len(count_iteration)):
			if hit_target[i]:
				LL_k += harmonic_sum(count_iteration[i])
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



def ibs_early_stopping_sequential(list_carlist, user_choice, inparams): # parallel computing
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

	positions = pickle.load(open('/Users/chloe/Desktop/carlists/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9_positions.pickle', 'rb'))[:500]
	decisions = pickle.load(open('/Users/chloe/Desktop/carlists/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9_decisions.pickle', 'rb'))[:500]

	pool = mp.Pool(processes=mp.cpu_count())
	guess = np.array([0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.1,6,0.1])

	def MLERegression(params):
		return ibs_early_stopping(positions, decisions, params, pool)
		# return my_ll_sequential(positions, decisions, params)

	results = minimize(MLERegression, guess, method = 'Nelder-Mead',
						options={'disp': True})
	print(results)
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



