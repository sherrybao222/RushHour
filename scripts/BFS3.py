'''
BFS model, speeded version, py27
Mainly for preparing movies for Weiji's talk at Brown Oct 16, 2019
    - Movie1: 
    	'predicted move is xxx, the completed decision tree is xxx.'
    - Movie2: 
    	multiple simulations, 
    	model's considerations, 
    	histogram and heatmap of each possible move 
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

import scipy.stats as stats
import imageio
from pprint import pprint
import time
import datetime
import cv2
import datetime
from matplotlib.ticker import NullFormatter


class Node:

	def __init__(self, cl):
		self.__car_list = cl
		self.__children = []
		self.__value = None
		self.__value_bylevel = None
		self.__tag = None # parent car to be moved
		self.__pos1from = None # parent position
		self.__pos2from = None # parent position 
		self.__pos1to = None # self position as a child
		self.__pos2to = None # self position as a child
		self.__board = None
		self.__red = None
		self.__orientation = None
		self.__length = None 

	def add_child(self, n):
		n.set_parent(self)
		self.__children.append(n)

	def move_from_parent(self, tag, pos1from, pos2from, pos1to, pos2to, orientation, length):
		''' if current node is a child, record how it is transformed from its parent 
		'''
		self.__tag = tag # car moved from parent
		self.__pos1from = pos1from # parent car position
		self.__pos2from	= pos2from # parent car position
		self.__pos1to = pos1to # current/child position
		self.__pos2to = pos2to # current/child position
		self.__orientation = orientation
		self.__length = length

	def get_move_from_parent(self):
		return self.__tag, self.__pos1from, self.__pos2from, self.__pos1to, self.__pos2to, self.__orientation, self.__length
		
	def get_move(self):
		return self.__tag, self.__pos1, self.__pos2, self.__pos1old, self.__pos2old

	def set_parent(self, p):
		self.__parent = p

	def set_value(self, v):
		self.__value = v

	def get_carlist(self):
		return self.__car_list

	def get_red(self):
		for car in self.__car_list:
			if car.tag == 'r':
				return car

	def get_board(self):
		tmp_b, _ = MAG.construct_board(self.__car_list)
		return tmp_b

	def get_value(self):
		if self.__value == None:
			# self.__value, self.__value_bylevel = Value1(self.__car_list, self.get_red())
			self.__value = Value1(self.__car_list, self.get_red())
		return self.__value

	def get_value_bylevel(self):
		return self.__value_bylevel

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

	def clean_children(self):
		self.__children = []

	def get_parent(self):
		return self.__parent
	
	def remove_child(self, c):
		for i in range(len(self.__children)):
			if self.__children[i] == c:
				c.parent = None
				self.__children.pop(i)
				return	

	def update_carlits(self, cl):
		self.__car_list = cl
		self.__board, self.__red = MAG.construct_board(self.__car_list)
		# self.__value, self.__value_bylevel = Value1(self.__car_list, self.__red)		
		self.__value = Value1(self.__car_list, self.__red)		

	def print_children(self):
		print('total number of children: '+str(len(self.__children)))
		for i in range(2):
			print('print first two children: '+str(i))
			print(MAG.board_to_str(self.__children[i].get_board()))

	def board_to_str(self):
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
		return out_str

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
	noise = np.random.normal(loc=mu, scale=sigma)
	# initialize MAG
	my_board2, my_red2 = MAG.construct_board(car_list2)
	new_car_list2 = MAG.construct_mag(my_board2, my_red2)
	# each following level
	new_car_list2 = MAG.assign_level(new_car_list2, my_red2)
	value = np.sum(np.array(MAG.get_num_cars_from_levels(new_car_list2, num_parameters)) * np.array(weights))
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
			for car in root_node.get_carlist():
				if car.tag == tag:
					pos1from, pos2from, orientation, length = car.start[0], car.start[1], car.orientation, car.length
			dummy_child = Node(new_list)
			dummy_child.move_from_parent(tag, pos1from, pos2from, pos[0], pos[1], orientation, length)
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
		# print('Random move made')
		return RandomMove(root), [], [], 'Random Move Made'
	else:
		reason = 'Model Stopped' # reason of termination
		DropFeatures(delta)
		if plot_flag:
			considered_node = [] # nodes already traversed along the branch in this ieration
		considered_node2 = [] # new node expanded along the branch in this iteration
		while not Stop(probability=gamma) and not Determined(root):
			n, traversed = SelectNode(root)
			# n, _ = SelectNode(root)
			if plot_flag:
				considered_node.append(traversed)
			n2 = ExpandNode(n, theta)
			considered_node2.append(n2)
			# Backpropagate(n, root)
			if n2.get_value() >= abs(np.random.normal(loc=mu, scale=sigma)): 
			# terminate the algorithm if found a terminal node
				reason = 'Solution Found'
				break
	if plot_flag:
		return ArgmaxChild(root), considered_node, considered_node2, reason
	return ArgmaxChild(root), [], considered_node2, reason

def estimate_prob(root_node, expected_board='', iteration=100):
	''' Estimate the probability of next possible moves given the root node '''
	# InitializeChildren(root_node)
	first_iteration = True
	frequency = None
	sol_idx = None
	
	for i in range(iteration):
		new_node, _, _, _ = MakeMove(root_node)
		if first_iteration:
			frequency = [0] * len(root_node.get_children())
			first_iteration = False
		child_idx = root_node.find_child(new_node)
		frequency[child_idx] += 1
	
	frequency = np.array(frequency, dtype=np.float32)/iteration # now frequency becomes probability

	for i in range(len(root_node.get_children())):
		if root_node.get_child(i).board_to_str() == expected_board:
			sol_idx = i

	return root_node.get_children(), frequency, sol_idx, [], []




def plot_tree():
	''' Show movie of the model's consideration of next moves 
		based on subject's current position from a trial. 
		Also plot the decision tree
	'''
	os.chdir(dir_name)
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1 # move number in this human trial
	img_count = 1

	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		
		# plot text
		# plot_blank(instance, img_count, text='Move Number '+str(move_num), color='orange')
		# img_count += 1
		# plot blank space
		plot_blank(instance, img_count, text='Initial Board', color='orange')
		img_count += 1
		# plot initial board
		plot_state(cur_node, instance, img_count)
		img_count += 1

		# initialize tree plot if required
		dot = Digraph(format='jpg', strict=True)
		dot.attr(size='12.8,9.6', fixedsize='true', 
				ratio='fill', margin='1, 0.5')
		dot.node(str(id(cur_node)), 
				str(format(cur_node.get_value(), '.2f')),
				fixedsize='true', width='1', height='0.75', 
				fontsize='25', penwidth='1.3')
		
		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']

		# run model to make one decision (which contains many iterations)
		selectedmove, considered, considered2, reason = MakeMove(cur_node)
		total_iteration = len(considered2)
		cur_iteration_num = 1 # initialize iteration count

		if considered == []:
			# plot text
			plot_blank(instance, img_count, text='Random Move', color='green')
			img_count += 1
		# plot each itertaion seperately
		for pos, pos2 in zip(considered, considered2): # if any iteration is considered
			# plot the node traversed along the selected branch in this iteration
			tree_cur = cur_node
			for pos_cur in pos: 
				for child in tree_cur.get_children():
					dot.node(str(id(child)), 
						str(format(child.get_value(), '.2f')),
						fixedsize='true', width='1', height='0.75',
						fontsize='25', penwidth='1.3')
					dot.edge(str(id(tree_cur)), str(id(child)), penwidth='1.3')
				tree_cur = pos_cur
			# plot the new node expanded along this branch in this iteration
			for child in tree_cur.get_children():
				dot.node(str(id(child)), 
					str(format(child.get_value(), '.2f')),
					fixedsize='true', width='1', height='0.75',
					fontsize='25', penwidth='1.3')
				dot.edge(str(id(tree_cur)), str(id(child)), penwidth='1.3')
			tree_cur = pos2
			for child in tree_cur.get_children():
				dot.node(str(id(child)), 
					str(format(child.get_value(), '.2f')),
					fixedsize='true', width='1', height='0.75', 
					fontsize='25', penwidth='1.3')
				dot.edge(str(id(tree_cur)), str(id(child)), penwidth='3')

		# plot selected move 
		plot_blank(instance, img_count, text='Selected Move', color='green')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1
		plot_state(selectedmove, instance, img_count)
		img_count += 1

		# make human move, mark human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		human_node = cur_node.get_child(cur_node.find_child_by_str(Node(cur_carlist).board_to_str()))
		dot.node(str(id(human_node)), 
				str(format(human_node.get_value(), '.2f')),
				fixedsize='true', width='1', height='0.75',
				fontsize='25', penwidth='1.3')
		dot.edge(str(id(cur_node)), 
				str(id(human_node)), 
				color='orange', 
				label='human', fontsize='25',
				penwidth='3')

		# prepare to plot actual move made by human 
		plot_blank(instance, img_count, text='Human Move', color='red')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1
		
		# mark model decision
		dot.node(str(id(selectedmove)), 
			str(format(selectedmove.get_value(), '.2f')),
			fixedsize='true', width='1', height='0.75',
			fontsize='25', penwidth='1.3')
		dot.edge(str(id(cur_node)), 
					str(id(selectedmove)), 
					color='red',
					label='model', fontsize='25', 
					penwidth='3')

		# update current node
		cur_node = Node(cur_carlist)
		# plot actual move made by human
		plot_state(cur_node, instance, img_count)
		img_count += 1

		# prepare to plot tree into movie
		plot_blank(instance, img_count, text='Decision Tree', color='green')
		img_count += 1
		# plot tree
		dot.render('/Users/chloe/Desktop/RHfig/'+instance+'_'+str(move_num)+'_tree', 
					view=False)

		# increment paramter
		move_num += 1

		# make movie
		make_movie(move_num-1, imgtype='tree')
		
		# clean all jpg files after movie done
		test = os.listdir(dir_name)
		for item in test:
		    if item.endswith(".jpg") or item.endswith('tree'):
		        os.remove(os.path.join(dir_name, item))



def plot_model():
	''' 		
		Movie of multiple simulations, 
		model's considerations and decisions,
		histogram and heatmap of each possible move,
		based on subject's current position from a trial. 
	'''
	os.chdir(dir_name)
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1 # move number in this human trial
	img_count = 1

	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		
		# plot text
		# plot_blank(instance, img_count, text='Move Number '+str(move_num), color='orange')
		# plot blank space
		plot_blank(instance, img_count, text='Initial Board', color='orange')
		img_count += 1
		# plot initial board
		plot_state(cur_node, instance, img_count)
		img_count += 1

		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']

		# run model to make one decision (which contains many iterations)
		selectedmove, considered, considered2, reason = MakeMove(cur_node)
		total_iteration = len(considered2)
		cur_iteration_num = 1 # initialize iteration count

		if considered == []:
			# plot text
			plot_blank(instance, img_count, text='Random Move', color='green')
			img_count += 1
		# plot each itertaion seperately
		for pos, pos2 in zip(considered, considered2): # if any iteration is considered
			# plot text, show/notshow the total number of iterations
			# plot_blank(instance, img_count, text='Iteration '+str(cur_iteration_num)+'/'+str(total_iteration), color='blue')
			plot_blank(instance, img_count, text='Iteration '+str(cur_iteration_num), color='blue')
			cur_iteration_num += 1
			img_count += 1
			# plot board
			plot_state(cur_node, instance, img_count) # initial state
			img_count += 1
			# plot the node traversed along the selected branch in this iteration
			tree_cur = cur_node
			for pos_cur in pos:
				# plot traversed nodes along the best branch
				plot_state(pos_cur, instance, img_count) 
				img_count += 1
			# plot the new node expanded along this branch in this iteration
			plot_state(pos2, instance, img_count)
			img_count += 1

		# plot selected move 
		plot_blank(instance, img_count, text=reason, color='green')
		img_count += 1
		plot_blank(instance, img_count, text='Selected Move', color='green')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1
		plot_state(selectedmove, instance, img_count)
		img_count += 1

		# make human move, mark human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)

		# estimate probability through many simulations
		children_list, frequency, sol_idx, _, _ = estimate_prob(cur_node, expected_board=Node(cur_carlist).board_to_str())
		
		# plot histogram
		barlist = plt.bar(np.arange(len(frequency)), frequency)
		barlist[sol_idx].set_color('r')
		plt.ylim(top=1.0, bottom=0)
		plt.title('Instance '+instance+', move number '+str(move_num))
		plt.savefig(dir_name+instance+'_hist_move_'+str(move_num)+'.jpg')
		plt.close()

		# plot heatmap
		plot_heatmap(cur_node, instance, move_num, children_list, frequency, sol_idx, imgtype='heatmap')
		img_count += 1

		# prepare to plot actual move made by human 
		plot_blank(instance, img_count, text='Human Move', color='red')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1
		
		# update current node
		cur_node = Node(cur_carlist)
		# plot actual move made by human
		plot_state(cur_node, instance, img_count)
		img_count += 1

		# prepare to plot histogram and heatmap
		plot_blank(instance, img_count, text='P(Human Move):\n'+str(format(frequency[sol_idx], '.3f')), color='orange')
		img_count += 1
		plot_blank(instance, img_count, text='Heatmap', color='orange')
		img_count += 1

		# increment paramter
		move_num += 1

		# make movie
		make_movie(move_num-1, imgtype='model')
		
		# clean all jpg files after movie done
		test = os.listdir(dir_name)
		for item in test:
		    if item.endswith(".jpg") or item.endswith('tree'):
		        os.remove(os.path.join(dir_name, item))


def plot_tree_evolution():
	''' 		
		Movie of multiple simulations, 
		model's considerations and decisions and tree evolution.
		histogram and heatmap of each possible move,
		based on subject's current position from a trial. 
	'''
	os.chdir(dir_name)
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	move_num = 1 # move number in this human trial
	img_count = 1

	# every human move in the trial
	for i in range(trial_start-1, trial_end-2):
		# plot text
		# plot_blank(instance, img_count, text='Move Number '+str(move_num), color='orange')
		# plot blank space
		plot_blank(instance, img_count, text='Initial Board', color='orange')
		img_count += 1
		# plot initial board
		plot_state(cur_node, instance, img_count)
		img_count += 1

		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']

		# run model to make one decision (which contains many iterations)
		selectedmove, considered, considered2, reason = MakeMove(cur_node)
		total_iteration = len(considered2)
		cur_iteration_num = 1 # initialize iteration count

		if considered == []:
			# plot text
			plot_blank(instance, img_count, text='Random Move', color='green')
			img_count += 1
		# plot each itertaion seperately
		for pos, pos2 in zip(considered, considered2): # if any iteration is considered	
			# plot text, show/notshow the total number of iterations
			# plot_blank(instance, img_count, text='Iteration '+str(cur_iteration_num)+'/'+str(total_iteration), color='blue')
			plot_blank(instance, img_count, text='Iteration '+str(cur_iteration_num), color='blue')
			cur_iteration_num += 1
			img_count += 1
			# plot board
			plot_state(cur_node, instance, img_count) # initial state
			img_count += 1
			# plot the node traversed along the selected branch in this iteration
			tree_cur = cur_node
			# initialize tree plot 
			dot = Digraph(format='jpg', strict=True)
			dot.attr(size='12.8,9.6', fixedsize='true', 
					ratio='fill', margin='1, 0.5')
			dot.node(str(id(cur_node)), 
					str(format(cur_node.get_value(), '.2f')),
					fixedsize='true', width='1', height='0.75', 
					fontsize='25', penwidth='1.3')
			for pos_cur in pos: 
				# plot traversed nodes along the best branch
				plot_state(pos_cur, instance, img_count) 
				img_count += 1
				for child in tree_cur.get_children():
					dot.node(str(id(child)), 
						str(format(child.get_value(), '.2f')),
						fixedsize='true', width='1', height='0.75',
						fontsize='25', penwidth='1.3')
					dot.edge(str(id(tree_cur)), str(id(child)), penwidth='1.3')
				tree_cur = pos_cur
			# plot the new node expanded along this branch in this iteration
			plot_state(pos2, instance, img_count)
			img_count += 1
			# plot the new node expanded along this branch in this iteration
			for child in tree_cur.get_children():
				dot.node(str(id(child)), 
					str(format(child.get_value(), '.2f')),
					fixedsize='true', width='1', height='0.75',
					fontsize='25', penwidth='1.3')
				dot.edge(str(id(tree_cur)), str(id(child)), penwidth='1.3')
			tree_cur = pos2
			for child in tree_cur.get_children():
				dot.node(str(id(child)), 
					str(format(child.get_value(), '.2f')),
					fixedsize='true', width='1', height='0.75', 
					fontsize='25', penwidth='1.3')
				dot.edge(str(id(tree_cur)), str(id(child)), penwidth='1.3')
			dot.render('/Users/chloe/Desktop/RHfig/'+instance+'_'+str(img_count)+'_board',
					view=False)
			img_count += 1

		# plot selected move 
		plot_blank(instance, img_count, text=reason, color='green')
		img_count += 1
		plot_blank(instance, img_count, text='Selected Move', color='green')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1
		plot_state(selectedmove, instance, img_count)
		img_count += 1

		# make human move, mark human move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		# estimate probability through many simulations
		children_list, frequency, sol_idx, _, _ = estimate_prob(cur_node, expected_board=Node(cur_carlist).board_to_str())
		
		# plot histogram
		barlist = plt.bar(np.arange(len(frequency)), frequency)
		barlist[sol_idx].set_color('r')
		plt.ylim(top=1.0, bottom=0)
		plt.title('Instance '+instance+', move number '+str(move_num))
		plt.savefig(dir_name+instance+'_hist_move_'+str(move_num)+'.jpg')
		plt.close()

		# plot heatmap
		plot_heatmap(cur_node, instance, move_num, children_list, frequency, sol_idx, imgtype='heatmap')
		img_count += 1

		# prepare to plot actual move made by human 
		plot_blank(instance, img_count, text='Human Move', color='red')
		img_count += 1
		plot_state(cur_node, instance, img_count)
		img_count += 1
		
		# update current node
		cur_node = Node(cur_carlist)
		# plot actual move made by human
		plot_state(cur_node, instance, img_count)
		img_count += 1

		# prepare to plot histogram and heatmap
		plot_blank(instance, img_count, text='P(Human Move):\n'+str(format(frequency[sol_idx], '.3f')), color='orange')
		img_count += 1
		plot_blank(instance, img_count, text='Heatmap', color='orange')
		img_count += 1

		# increment paramter
		move_num += 1
		# make movie
		# make_movie(move_num-1, imgtype='model')
		
		# clean all jpg files after movie done
		test = os.listdir(dir_name)
		for item in test:
		    if item.endswith("board") or item.endswith('tree'):
		        os.remove(os.path.join(dir_name, item))

		sys.exit()


def plot_trial_human(trial_start=2, trial_end=20):
	''' show movie of a human trial
		trial_start: starting row number in the raw data
		trial_end: ending row number in the raw data
	'''
	sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')
	dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
	os.chdir(dir_name)
	# construct initial node
	first_line = sub_data.loc[trial_start-2,:]
	instance = first_line['instance']
	ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	ins_file = ins_dir + instance + '.json'
	initial_car_list, initial_red = MAG.json_to_car_list(ins_file)
	initial_board, initial_red = MAG.construct_board(initial_car_list)
	initial_node = Node(initial_car_list)
	# print('Initial board:\n'+initial_node.board_to_str())
	
	# initialize parameters
	cur_node = initial_node
	cur_carlist = initial_car_list
	global move_num
	move_num = 1
	img_count = 0 # image count

	# plot blank space
	plot_blank(instance, img_count, text='Human Trial, \n'+instance, color='orange', imgtype='human')
	img_count += 1
	# plot initial board
	plot_state(cur_node, instance, img_count, imgtype='human')
	img_count += 1

	# every human move in the trial
	for i in range(trial_start-1, trial_end-1):
	
		# load data from datafile
		print('Move number '+str(move_num))
		row = sub_data.loc[i, :]
		piece = row['piece']
		move_to = row['move']

		# make move
		cur_carlist, _ = MAG.move(cur_carlist, piece, move_to)
		cur_board, _ = MAG.construct_board(cur_carlist)
		cur_node = Node(cur_carlist)
		move_num += 1
		
		# plot current human move 
		plot_state(cur_node, instance, img_count, imgtype='human')
		img_count += 1

	# make movie and save
	make_movie(1, format='avi', imgtype='human')
	
	# clean all jpg files after movie done
	test = os.listdir(dir_name)
	for item in test:
	    if item.endswith("human.jpg"):
	        os.remove(os.path.join(dir_name, item))
	





def str_to_matrix(string):
	''' convert string of board to a int matrix '''
	matrix = np.zeros((6,6), dtype=int)
	line_idx = 0
	for line in string.split('\n'):
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
	return matrix

def plot_state(cur_node, instance, idx, out_file='/Users/chloe/Desktop/RHfig/', imgtype='board'):
	''' visualize the current node/board configuration '''
	matrix = str_to_matrix(cur_node.board_to_str())
	matrix = np.ma.masked_where(matrix==-1, matrix)
	cmap = plt.cm.Set1
	cmap.set_bad(color='white')
	fig, ax = plt.subplots(figsize=(12.8,9.6))
	ax.set_xticks(np.arange(-0.5, 5, 1))
	ax.set_yticks(np.arange(-0.5, 5, 1))
	ax.set_axisbelow(True)
	ax.grid(b=True, which='major',color='gray', linestyle='-', linewidth=1, alpha=0.1)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	for tic in ax.xaxis.get_major_ticks():
		tic.tick1On = tic.tick2On = False
	for tic in ax.yaxis.get_major_ticks():
		tic.tick1On = tic.tick2On = False
	im = ax.imshow(matrix, cmap=cmap)
	show_car_label = False
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
	if imgtype == 'human':
		plt.title('Move '+str(move_num-1), fontsize=26)
	# ax.tick_params(axis='both', which='major', labelsize=25)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(4)
		ax.spines[axis].set_zorder(0)
	fig.patch.set_facecolor('grey')
	fig.patch.set_alpha(0.2)
	plt.savefig(out_file+instance+'_'+str(idx)+'_'+imgtype+'.jpg', 
			facecolor = fig.get_facecolor(), transparent = True)
	plt.close()

def plot_blank(instance, idx, text, color, out_file='/Users/chloe/Desktop/RHfig/', imgtype='board'):
	''' plot a blank image
		with the entire page filled by one color and a text message '''
	fig, ax = plt.subplots(figsize=(12.8,9.6))
	fig.patch.set_facecolor(color)
	fig.patch.set_alpha(0.1)
	ax.patch.set_facecolor(color)
	ax.patch.set_alpha(0.1)
	ax.text(0.3, 0.5, text, fontsize=40)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.axis('off')
	fig.savefig(out_file+instance+'_'+str(idx)+'_'+imgtype+'.jpg', \
				facecolor=fig.get_facecolor(), edgecolor='none')
	plt.close()

def plot_heatmap(cur_node, instance, move_num, children_list, frequency, sol_idx, out_file='/Users/chloe/Desktop/RHfig/', imgtype='heatmap'):
	''' visualize the arrows of heatmap  '''
	matrix = str_to_matrix(cur_node.board_to_str())
	matrix = np.ma.masked_where(matrix==-1, matrix)
	cmap = plt.cm.Set1
	cmap.set_bad(color='white')
	fig, ax = plt.subplots()
	ax.set_xticks(np.arange(-0.5, 5, 1))
	ax.set_yticks(np.arange(-0.5, 5, 1))
	ax.set_axisbelow(True)
	ax.grid(b=True, which='major',color='gray', linestyle='-', linewidth=1, alpha=0.1)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	for tic in ax.xaxis.get_major_ticks():
		tic.tick1On = tic.tick2On = False
	for tic in ax.yaxis.get_major_ticks():
		tic.tick1On = tic.tick2On = False
	im = ax.imshow(matrix, cmap=cmap)
	show_car_label = False
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
				text = ax.text(j, i, num, ha="center", va="center", color="black", fontsize=14)
	count = 0
	for child in children_list:
		tag, pos1from, pos2from, pos1to, pos2to, orientation, length = child.get_move_from_parent()
		if count == sol_idx:
			if frequency[count] != 0:
				if orientation == 'horizontal' and (pos1to-pos1from)>0:
					plt.arrow(x=pos1from+length-1, y=pos2from, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
						head_width=0.20, head_length=0.15, alpha=0.5, color='red',
						lw= 25 * frequency[count])
				elif orientation == 'vertical' and (pos2to-pos2from)>0:
					plt.arrow(x=pos1from, y=pos2from+length-1, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
						head_width=0.20, head_length=0.15, alpha=0.5, color='red',
						lw= 25 * frequency[count])
				else:
					plt.arrow(x=pos1from, y=pos2from, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
						head_width=0.20, head_length=0.15, alpha=0.5, color='red',
						lw= 25 * frequency[count])
		else:
			if frequency[count] != 0:
				if orientation == 'horizontal' and (pos1to-pos1from)>0:
					plt.arrow(x=pos1from+length-1, y=pos2from, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
						head_width=0.20, head_length=0.15, alpha=0.5, color='black',
						lw= 25 * frequency[count])
				elif orientation == 'vertical' and (pos2to-pos2from)>0:
					plt.arrow(x=pos1from, y=pos2from+length-1, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
						head_width=0.20, head_length=0.15, alpha=0.5, color='black',
						lw= 25 * frequency[count])
				else:
					plt.arrow(x=pos1from, y=pos2from, dx=(pos1to-pos1from), dy=(pos2to-pos2from), 
						head_width=0.20, head_length=0.15, alpha=0.5, color='black',
						lw= 25 * frequency[count])
		count += 1
	# plt.title('Heatmap Move '+str(move_num))
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(4)
		ax.spines[axis].set_zorder(0)
	fig.patch.set_facecolor('grey')
	fig.patch.set_alpha(0.2)
	plt.savefig(out_file+instance+'_'+imgtype+'_move_'+str(move_num)+'.jpg',
				facecolor = fig.get_facecolor(), transparent = True)
	plt.close()

def make_movie(move_num, path='/Users/chloe/Desktop/RHfig/', format='avi', imgtype='tree'):
	''' make a movie using jpg files '''
	os.chdir(path)
	if imgtype == 'tree':
		image_folder = path
		video_name = 'MOVIE-'+imgtype+'-%s.avi' % (str(move_num)+'-'+datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
		images = sorted(filter(os.path.isfile, [x for x in os.listdir(path) if x.endswith('_board.jpg')]), key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))
		images.append(instance+'_'+str(move_num)+'_tree.jpg')
		# print(images)
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape
		video = cv2.VideoWriter(video_name, 0, 1, (width,height))
		for image in images:
			resized=cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height)) 
			video.write(resized)
		cv2.destroyAllWindows()
		video.release()
	elif imgtype == 'model':
		image_folder = path
		video_name = 'MOVIE-'+imgtype+'-%s.avi' % (str(move_num)+'-'+datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
		images = sorted(filter(os.path.isfile, [x for x in os.listdir(path) if x.endswith('_board.jpg')]), key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))
		# images.append(instance+'_hist_move_'+str(move_num)+'.jpg')
		images.append(instance+'_heatmap_move_'+str(move_num)+'.jpg')
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape
		video = cv2.VideoWriter(video_name, 0, 1, (width,height))
		for image in images:
			resized=cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height)) 
			video.write(resized)
		cv2.destroyAllWindows()
		video.release()
	elif imgtype == 'evolve':
		image_folder = path
		video_name = 'MOVIE-'+imgtype+'-%s.avi' % (str(move_num)+'-'+datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
		images = sorted(filter(os.path.isfile, [x for x in os.listdir(path) if x.endswith('jpg')]))
		# print images
		# sys.exit()
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape
		video = cv2.VideoWriter(video_name, 0, 1, (width,height))
		for image in images:
			resized=cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height)) 
			video.write(resized)
		cv2.destroyAllWindows()
		video.release()
	elif imgtype == 'human':
		image_folder = path
		video_name = 'MOVIE-'+imgtype+'-%s.avi' % (str(move_num)+'-'+datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S'))
		images = sorted(filter(os.path.isfile, [x for x in os.listdir(path) if x.endswith('_human.jpg')]), key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape
		video = cv2.VideoWriter(video_name, 0, 1, (width,height))
		for image in images:
			resized=cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height)) 
			video.write(resized)
		cv2.destroyAllWindows()
		video.release()

def filename(x):
    return x[:20]


global w0, w1, w2, w3, w4, w5, w6, w7, mu, sigma, weights, num_parameters
w0 = 0
w1 = -7
w2 = -10
w3 = -7
w4 = -6
w5 = -5
w6 = -4
w7 = -1
mu = 0
sigma = 1
weights = [w0, w1, w2, w3, w4, w5, w6, w7]
num_parameters = len(weights)
trial_start = 2 # starting row number in the raw data, 21,53
trial_end = 20
global move_num # move number in this human trial
global plot_flag # whether to plot
plot_flag = True
sub_data = pd.read_csv('/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv')

# construct initial node
first_line = sub_data.loc[trial_start-2,:]
instance = first_line['instance']
ins_dir = '/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
dir_name = '/Users/chloe/Desktop/RHfig/' # dir for new images
ins_file = ins_dir + instance + '.json'
initial_car_list, _ = MAG.json_to_car_list(ins_file)
initial_node = Node(initial_car_list)
print('========================== started =======================')




# plot_tree()
# plot_model()
# plot_trial_human()
# plot_tree_evolution()
make_movie(1, imgtype='evolve')


