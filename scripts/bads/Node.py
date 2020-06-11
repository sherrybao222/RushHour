from Board import *
from MAG import *
import numpy as np
import time

class Node:
	def __init__(self, board, mag, params, parent=None):
		self.board = board # Board
		self.mag = mag
		self.parent = parent # Node
		self.children = [] # list of Node
		self.params = params
		if self.mag != None:
			self.value = - (np.sum(np.array(self.mag.num_cars_each_level, dtype=np.int64) 
					* np.array(self.params.weights, dtype=np.float64), dtype=np.float64)
					+ np.random.normal(loc=self.params.mu, scale=self.params.sigma))
		else:
			self.value=None
	def __members(self):
		return (self.board, self.parent, self.children, self.value, self.params)
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
			if c.board.print_board() == bstr:
				return c
		return None
	def remove_child(self, idx):
		return self.children.pop(idx)
	def heuristic_value(self):
		'''
		value = (-1) * [w0 * num_cars{MAG-level RED}
			+ w1 * num_cars{MAG-level 1} 
			+ w2 * num_cars{MAG-level 2}  
			+ w3 * num_cars{MAG-level 3} 
			+ w4 * num_cars{MAG-level 4} 
			+ w5 * num_cars{MAG-level 5} 
			+ w6 * num_cars{MAG-level 6}
			+ w7 * num_cars{MAG-level 7}  
			+ noise]
		weights are positive numbers
		value is negative
		value the larger/closer to 0 the better
		'''
		if self.value == None:
			v = np.sum(np.array(self.mag.num_cars_each_level, dtype=np.int64) 
					* np.array(self.params.weights, dtype=np.float64), dtype=np.float64)
			self.value = -(v+np.random.normal(loc=self.params.mu, scale=self.params.sigma))
		
		return self.value
