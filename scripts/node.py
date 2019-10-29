import MAG
# import random, sys, copy, os, pickle
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.optimize import minimize
import cProfile, pstats, StringIO
from operator import methodcaller
# import multiprocessing as mp
# from numpy import recfromcsv

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
