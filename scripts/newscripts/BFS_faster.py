from Board import *
from MAG import *
from Node import *
from operator import attrgetter
import pandas as pd
import random, pickle
from collections import OrderedDict


class Params:
	def __init__(self, w1, w2, w3, w4, w5, w6, w7, 
					stopping_probability,
					pruning_threshold,
					lapse_rate,
					puzzle,
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
		self.puzzle = puzzle


class LRUCache:
	'''
	least recent used (will be poped) cache
	each cache is specified for one puzzle only
	'''
	def __init__(self, puzzle, capacity=3000):
		self.cache = OrderedDict()
		self.puzzle = puzzle
		self.capacity = capacity
	def get(self, board_id):
		if board_id not in self.cache:
			return None
		else:
			self.cache.move_to_end(board_id)
			return self.cache[board_id]
	def put(self, board_id, pickle_object):
		self.cache[board_id] = pickle_object
		self.cache.move_to_end(board_id)
		if len(self.cache) > self.capacity:
			self.cache.popitem(last=False)


def DropFeatures(probability):
	pass

def Lapse(probability):
	''' return true with a probability '''
	return random.random() < probability

def Stop(probability): 
	''' return true with a probability '''
	return random.random() < probability

def RandomMove(node, params, cache):
	''' make a random move and return the resulted node '''
	assert not is_solved(node.board), "RandomMove input node is already solved."
	board_id = make_id(node.board)
	if cache.get(board_id)==None:
		cache.put(board_id, pickle.load(open(os.path.join(preoprocessed_data_path, params.puzzle, board_id)+'.p', 'rb')))
	all_children = cache.get(board_id)
	for i in range(len(all_children['children_boards'])):
		child = Node(all_children['children_boards'][i], parent=node)
		node.children.append(child)
	return random.choice(node.children)

def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	while len(n.children) != 0:
		n = n.maxchild
	return n, is_solved(n.board)
 
def ExpandNode(node, params, cache):
	''' 
	load all possible nodes under input node, 
	cut the ones below threshold 
	load positions from cache
	'''
	board_id = make_id(node.board)
	if cache.get(board_id)==None:
		cache.put(board_id, pickle.load(open(os.path.join(preoprocessed_data_path, params.puzzle, board_id)+'.p', 'rb')))
	all_children = cache.get(board_id)
	maxchild = None
	for i in range(len(all_children['children_boards'])):
		child = Node(all_children['children_boards'][i], all_children['children_num_cars_each_level'][i], params, parent=node)
		node.children.append(child)
		if maxchild == None or child.value > maxchild.value:
			node.maxchild = child
	Vmaxchild = node.maxchild
	for child_idx in range(len(node.children))[::-1]: # iterate in reverse order
		if abs(node.children[child_idx].value - Vmaxchild.value) > params.pruning_threshold:
			node.remove_child(child_idx)
	
def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	try:
		maxchild = ArgmaxChild(this_node)
		this_node.value = maxchild.value
		this_node.maxchild = maxchild
	except:
		this_node.value = this_node.value
	if this_node != root_node:
		Backpropagate(this_node.parent, root_node)

def ArgmaxChild(node): 
	''' 
		return the child with max value 
	'''
	# return max(node.children, key=attrgetter('value'))
	return max(node.children, key=lambda node:node.value)

def MakeMove(root, params, cache):
	''' 
	`	returns an optimal move to make next 
		according to value function and current board position
	'''
	assert len(root.children) == 0
	# check if root node is already a winning position
	if is_solved(root.board):
		# move red car to position 16 if rootis a winning position
		board = move(root.board, 'r', 16)
		return Node(board)
	if Lapse(params.lapse_rate): # random move
		return RandomMove(root, params, cache)
	else:
		DropFeatures(params.feature_dropping_rate)
		while not Stop(probability=params.stopping_probability):
			leaf, leaf_is_solution = SelectNode(root)
			if leaf_is_solution:
				Backpropagate(leaf.parent, root)
				continue
			ExpandNode(leaf, params, cache)
			Backpropagate(leaf, root)
	if root.children == []: # if did not enter while loop at all
		ExpandNode(root, params, cache)
	return root.maxchild



if __name__ == '__main__':
	'''
	load positions as necessary and keep cache
	'''

	# initialize data 
	home_dir = '/Users/yichen/'
	instancepath=home_dir+'Documents/RushHour/exp_data/data_adopted/'
	subject_path = home_dir+'Desktop/subjects/' + random.choice(os.listdir(home_dir+'Desktop/subjects/'))
	df = pd.read_csv(subject_path)
	preoprocessed_data_path = home_dir+'Desktop/preprocessed_positions/'
	print('subject: '+str(subject_path))

	# initialize preloaded preprocessed positions dictionary 
	preprocessed_positions = {}
	puzzle_cache={}
	inparams = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 0.01]
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
					w4=inparams[3], w5=inparams[4], w6=inparams[5], 
					w7=inparams[6], 
					stopping_probability=inparams[7],
					pruning_threshold=inparams[8],
					lapse_rate=inparams[9], puzzle='')
	

	# fill in cache
	for idx, row in df.iterrows():
		# read current move data 
		if row['event'] == 'start':
			instance = row['instance']
			ins_file = instancepath+instance+'.json'
			board = json_to_board(ins_file)
			if params.puzzle != row['instance']:
				params.puzzle = row['instance']
				if puzzle_cache.get(params.puzzle)==None:
					cache=LRUCache(params.puzzle)
					puzzle_cache[params.puzzle]=cache
				else:
					cache=puzzle_cache[params.puzzle]
			continue
		if row['piece']=='r' and row['move']==16 and row['optlen']==1: # skip the winning moves
			continue
		# call makemove and record time data
		MakeMove(Node(board), params, cache)
		# change the board and perform the current move
		piece = str(row['piece'])
		move_to = int(row['move'])
		board = move(board, piece, move_to)


	# record time
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
					w4=inparams[3], w5=inparams[4], w6=inparams[5], 
					w7=inparams[6], 
					stopping_probability=inparams[7],
					pruning_threshold=inparams[8],
					lapse_rate=inparams[9], puzzle='')
	bfs_start = time.time()
	makemove_time = []
	# iterate over each move
	for idx, row in df.iterrows():
		# read current move data 
		if row['event'] == 'start':
			instance = row['instance']
			ins_file = instancepath+instance+'.json'
			board = json_to_board(ins_file)
			if params.puzzle != row['instance']:
				params.puzzle = row['instance']
				if puzzle_cache.get(params.puzzle)==None:
					cache=LRUCache(params.puzzle)
					puzzle_cache[params.puzzle]=cache
				else:
					cache=puzzle_cache[params.puzzle]
				# makemove_time.append([])
			continue
		if row['piece']=='r' and row['move']==16 and row['optlen']==1: # skip the winning moves
			continue
		# call makemove and record time data
		mag = MAG(board)
		mag.construct()
		# makemove
		makemove_start = time.time()
		MakeMove(Node(board), params, cache)
		makemove_time.append(time.time()-makemove_start)
		# change the board and perform the current move
		piece = str(row['piece'])
		move_to = int(row['move'])
		board = move(board, piece, move_to)
		
	print('total BFS time '+str(time.time()-bfs_start))
	print('subject: '+str(subject_path))
	# print('every MakeMove time: '+str(makemove_time))
	print('average MakeMove time: '+str(np.mean(makemove_time)))

