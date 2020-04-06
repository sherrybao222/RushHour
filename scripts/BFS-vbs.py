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
