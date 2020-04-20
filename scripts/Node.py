from Car import *
from Board import *
import numpy as np
#Zahy Notes
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
	def remove_child(self, idx):
		return self.children.pop(idx)
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
		value the larger the better
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


