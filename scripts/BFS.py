# BFS model
# py27
import MAG
import random

class Node:
	car_list = None
	red = None
	board = None
	mag = None
	value = 0
	def __init__(self, cl, r):
		self.car_list = cl
		self.red = r
		self.board, _ = MAG.construct_board(car_list)
		self.mag = MAG.construct_mag(self.board, self.red)


def Lapse(probability): # return true with probability
	return random.random() < probability

def Stop(probability): # return true with probability
	return random.random() < probability

def Determined(root_node): # true if win
	return MAG.check_win(root_node.board, root_node.red)

def RandomMove(node):
	all_moves = MAG.all_legal_moves(node.car_list, node.red, node.board)
	(car, pos) = random.choice(all_moves)
	new_list, new_red = MAG.move2(node.car_list, car.tag, pos[0], pos[1])
	new_node = Node(new_list, new_red)
	return new_node


def DropFeatures(delta):


def SelectNode():


def Backpropagate(n):


def argmaxChild(r):


def MakeMove(state):
	if Lapse():
		return RandomMove(state)
	else:
		DropFeatures(delta)
		root = Node(state)
		while not Stop(gamma) and not Determined(root):
			n = SelectNode()
			ExpandNode(n)
			Backpropagate(n)
	return argmaxChild(root)






