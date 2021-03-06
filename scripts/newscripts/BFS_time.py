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

class MRUCache:
	'''
	most recent used (will be poped) cache
	each cache is specified for one puzzle only
	deprecated
	'''
	def __init__(self, puzzle, capacity=1000):
		self.cache = OrderedDict()
		self.puzzle = puzzle
		self.capacity = capacity
	def get(self, board_id):
		if board_id not in self.cache:
			return None
		else:
			self.cache.move_to_end(board_id, last=False)
			return self.cache[board_id]
	def put(self, board_id, pickle_object):
		self.cache[board_id] = pickle_object
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

def RandomMove(node, params):
	''' make a random move and return the resulted node '''
	assert not is_solved(node.board), "RandomMove input node is already solved."
	all_boards = all_legal_moves(node.board)
	for new_board in all_boards:
		child = Node(new_board, None, params)
		child.parent = node
		node.children.append(child)
	return random.choice(node.children)

def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	while len(n.children) != 0:
		# n = ArgmaxChild(n)
		n = n.maxchild
	return n, is_solved(n.board)
 
def ExpandNode_12(node, params):
	''' 
	preload = 1 or 2
	create all possible nodes under input node, 
	cut the ones below threshold 
	'''
	InitializeChildren_start = time.time()
	all_children = preprocessed_positions[params.puzzle][make_id(node.board)]
	time_dict['load_preprocessed'].append(time.time() - InitializeChildren_start)
	for i in range(len(all_children['children_ids'])):
		childNode_start = time.time()	
		child = Node(all_children['children_boards'][i], all_children['children_mags'][i], params)
		child.parent = node
		node.children.append(child)
		time_dict['create_Node'].append(time.time() - childNode_start)
	time_dict['InitializeChildren'].append(time.time() - InitializeChildren_start)
	Vmaxchild = ArgmaxChild(node)
	for child_idx in range(len(node.children))[::-1]: # iterate in reverse order
		if abs(node.children[child_idx].value - Vmaxchild.value) > params.pruning_threshold:
			node.remove_child(child_idx)

def ExpandNode_0(node, params):
	''' 
	preload = 0
	create all possible nodes under input node, 
	cut the ones below threshold 
	loading every board position from file again
	'''
	InitializeChildren_start = time.time()
	all_children = pickle.load(open(os.path.join(preoprocessed_data_path, params.puzzle, make_id(node.board))+'.p', 'rb'))
	time_dict['load_preprocessed'].append(time.time() - InitializeChildren_start)
	for i in range(len(all_children['children_ids'])):
		childNode_start = time.time()	
		child = Node(all_children['children_boards'][i], all_children['children_mags'][i], params)
		child.parent = node
		node.children.append(child)
		time_dict['create_Node'].append(time.time() - childNode_start)
	Vmaxchild = ArgmaxChild(node)
	for child_idx in range(len(node.children))[::-1]: # iterate in reverse order
		if abs(node.children[child_idx].value - Vmaxchild.value) > params.pruning_threshold:
			node.remove_child(child_idx)

def ExpandNode(node, params, cache):
	''' 
	preload = 3
	create all possible nodes under input node, 
	cut the ones below threshold 
	load positions from cache
	'''
	InitializeChildren_start = time.time()
	board_id = make_id(node.board)
	if cache.get(board_id)==None:
		cache.put(board_id, pickle.load(open(os.path.join(preoprocessed_data_path, params.puzzle, board_id)+'.p', 'rb')))
	all_children = cache.get(board_id)
	time_dict['load_preprocessed'].append(time.time() - InitializeChildren_start)
	maxchild = None
	for i in range(len(all_children['children_ids'])):
		childNode_start = time.time()
		child = Node(all_children['children_boards'][i], all_children['children_mags'][i], params, parent=node)
		node.children.append(child)
		if maxchild == None or child.value > maxchild.value:
			node.maxchild = child
		time_dict['create_Node'].append(time.time() - childNode_start)
	# Vmaxchild = ArgmaxChild(node)
	Vmaxchild = node.maxchild
	for child_idx in range(len(node.children))[::-1]: # iterate in reverse order
		if abs(node.children[child_idx].value - Vmaxchild.value) > params.pruning_threshold:
			node.remove_child(child_idx)
	
def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	# this_node.value = ArgmaxChild(this_node).value
	maxchild = ArgmaxChild(this_node)
	this_node.value = maxchild.value
	this_node.maxchild = maxchild
	if this_node != root_node:
		Backpropagate(this_node.parent, root_node)

def ArgmaxChild(node): 
	''' 
		return the child with max value 
	'''
	ArgmaxChild_start = time.time()
	result = max(node.children, key=attrgetter('value'))
	time_dict['ArgmaxChild'].append(time.time() - ArgmaxChild_start)
	return result

def MakeMove(root, params, cache, hit=False, verbose=False):
	''' 
	`	returns an optimal move to make next 
		according to value function and current board position
	'''
	MakeMove_start = time.time()
	beforewhile_start = time.time()
	if hit: # for ibs, if matches human decision, return root node
		time_dict['MakeMove'].append(time.time()- MakeMove_start)
		return root
	assert len(root.children) == 0
	# check if root node is already a winning position
	if is_solved(root.board):
		# move red car to position 16 if rootis a winning position
		board = move(root.board, 'r', 16)
		result = Node(board, None, params)
		return result
	if Lapse(params.lapse_rate): # random move
		RandomMove_start = time.time()
		result = RandomMove(root, params)
		time_dict['beforewhile'].append(time.time() - beforewhile_start)
		time_dict['MakeMove'].append(time.time()- MakeMove_start)
		return result
	else:
		DropFeatures(params.feature_dropping_rate)
		Stop_start = time.time()
		should_stop = Stop(probability=params.stopping_probability)
		time_dict['Stop'].append(time.time() - Stop_start)
		time_dict['beforewhile'].append(time.time() - beforewhile_start)
		while_start = time.time()
		while not should_stop:
			SelectNode_start = time.time()
			leaf, leaf_is_solution = SelectNode(root)
			if verbose:
				print('\n\nnew while')
				print('Leaf Node is solution: '+str(leaf_is_solution))
				print('leaf Node\n'+str(leaf.board_to_str()))
				print('\n\n')
			time_dict['SelectNode'].append(time.time()- SelectNode_start)
			if leaf_is_solution:
				Backpropagate_start = time.time()
				Backpropagate(leaf.parent, root)
				time_dict['Backpropagate'].append(time.time() - Backpropagate_start)
				Stop_start = time.time()
				should_stop = Stop(probability=params.stopping_probability)
				time_dict['Stop'].append(time.time() - Stop_start)
				continue
			ExpandNode_start = time.time()
			ExpandNode(leaf, params, cache)
			if verbose:
				leaf.print_children()
			time_dict['ExpandNode'].append(time.time() - ExpandNode_start)
			Backpropagate_start = time.time()
			Backpropagate(leaf, root)
			time_dict['Backpropagate'].append(time.time() - Backpropagate_start)
			Stop_start = time.time()
			should_stop = Stop(probability=params.stopping_probability)
			time_dict['Stop'].append(time.time() - Stop_start)
		time_dict['insidewhile'].append(time.time() - while_start)
		afterwhile_start = time.time()
	if root.children == []: # if did not enter while loop at all
		ExpandNode_start = time.time()
		ExpandNode(root, params, cache)
		time_dict['ExpandNode'].append(time.time() - ExpandNode_start)
	# result = ArgmaxChild(root)
	result = root.maxchild
	time_dict['afterwhile'].append(time.time() - afterwhile_start)
	time_dict['MakeMove'].append(time.time()- MakeMove_start)
	return result


def preload_data(puzzle, preoprocessed_data_path):
	'''
	load preprocessed positions of one particular puzzle into a dictionary
	'''
	result = {}
	result[puzzle]={}
	print('loading puzzle ' + puzzle)
	for position_file in os.listdir(os.path.join(preoprocessed_data_path, puzzle)):
		if not position_file.endswith('.p'):
			continue
		position = position_file[:-2]
		result[puzzle][position] = pickle.load(open(os.path.join(preoprocessed_data_path,puzzle,position_file), 'rb'))
	return result




def harmonic_sum(n):
	''' 
		return sum of harmonic series from 1 to n-1
		when n=1, return 0
	'''
	s = 0.0
	for i in range(1, n):
		s += 1.0/i
	return s



if __name__ == '__main__':
	'''
	preload=0: no preload, load necessary positions in ExpandNode; 
	1: preload all positions at once; 
	2: preload one puzzle as needed;
	3: load positions as necessary and keep cache
	'''
	preload=3
	bfs_start = time.time()
	# initialize data 
	home_dir = '/Users/yichen/'
	all_datapath = home_dir+'Desktop/trialdata_valid_true_dist7_processed.csv'
	outputpath = home_dir+'Desktop/timedict_new.csv'
	instancepath=home_dir+'Documents/RushHour/exp_data/data_adopted/'
	subject_path = home_dir+'Desktop/subjects/' + random.choice(os.listdir(home_dir+'Desktop/subjects/'))
	df = pd.read_csv(subject_path)
	preoprocessed_data_path = home_dir+'Desktop/preprocessed_positions/'
	print('subject: '+str(subject_path))

	# initialize preloaded preprocessed positions dictionary 
	preprocessed_positions = {}
	
	# preload all positions
	if preload==1:
		for puzzle in os.listdir(preoprocessed_data_path):
			if not os.path.isdir(os.path.join(preoprocessed_data_path,puzzle)):
				continue
			preprocessed_positions[puzzle] = {}
			print('preloading puzzle '+puzzle)
			for position_file in os.listdir(os.path.join(preoprocessed_data_path,puzzle)):
				if not position_file.enndswith('.p'):
					continue
				position = position_file[:-2]
				preprocessed_positions[puzzle][position]=pickle.load(open(os.path.join(preoprocessed_data_path,puzzle,position_file),'rb'))
	if preload==3:
		puzzle_cache={}

	# initialize parameters
	inparams = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 0.01]
	all_time_dict = {'MakeMove':[], 
					'insidewhile':[], 'beforewhile':[], 'afterwhile':[],
					'Stop':[], 'SelectNode':[], 'Backpropagate':[], 'ArgmaxChild':[],
					'ExpandNode':[], 
						'load_preprocessed':[], 'create_Node': [], 							
					}
	total_load_time = 0

	time_dict = all_time_dict
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
					w4=inparams[3], w5=inparams[4], w6=inparams[5], 
					w7=inparams[6], 
					stopping_probability=inparams[7],
					pruning_threshold=inparams[8],
					lapse_rate=inparams[9], puzzle='')
	
	# iterate over each move
	for idx, row in df.iterrows():
		# if idx == 1000:
		# 	break
		# initialize record for current move
		time_dict = {}
		for key in all_time_dict.keys():
			time_dict[key] = []
		# read current move data 
		if row['event'] == 'start':
			# load new instance data
			instance = row['instance']
			ins_file = instancepath+instance+'.json'
			board = json_to_board(ins_file)
			if params.puzzle != row['instance']:
				params.puzzle = row['instance']
				print('switch puzzle at index '+str(idx))
				if preload == 2:
					preload_start = time.time()
					preprocessed_positions = preload_data(puzzle=row['instance'], preoprocessed_data_path=preoprocessed_data_path)
					preload_time = time.time()-preload_start
					print('preload time: '+str(preload_time))
					total_load_time += preload_time
				if preload==3:
					if puzzle_cache.get(params.puzzle)==None:
						cache=LRUCache(params.puzzle)
						puzzle_cache[params.puzzle]=cache
					else:
						cache=puzzle_cache[params.puzzle]
			continue
		if row['piece']=='r' and row['move']==16 and row['optlen']==1: # win
			# skip the winning moves
			continue

		# call makemove and record time data
		mag = MAG(board)
		mag.construct()
		MakeMove(Node(board, mag, params), params, cache)
		
		# change the board and perform the current move
		piece = str(row['piece'])
		move_to = int(row['move'])
		board = move(board, piece, move_to)
		
		# summarize all records from current move
		for key in all_time_dict.keys():
			all_time_dict[key].append(sum(time_dict[key]))

	print('subject: '+str(subject_path))
	if preload==2:
		print('total preload time '+str(total_load_time))
	print('total BFS time '+str(time.time()-bfs_start))
	# all moves completed, convert to dataframe
	all_time_df = pd.DataFrame.from_dict(all_time_dict)
	# save files as csv and calculate summary statistics
	all_time_df.to_csv(outputpath, index = False, header=True)
	df = pd.read_csv(outputpath)
	for col in df.columns[1:]:
		df[col] = df[col]/df['MakeMove']
	print(df.mean())

