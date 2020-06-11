from Board import *
from MAG import *
from Node import *
from operator import attrgetter
import pandas as pd
import random, pickle
from collections import OrderedDict
import multiprocessing as mp
import numpy as np
import matlab.engine


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


class LRUCache:
	'''
	least recent used (will be poped) cache
	each cache is specified for one puzzle only
	'''
	def __init__(self, puzzle, capacity=2000):
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
		n = ArgmaxChild(n)
	return n, is_solved(n.board)
 
def ExpandNode(node, params, cache, puzzle, preoprocessed_data_path='/Users/yichen/Desktop/preprocessed_positions/'):
	''' 
	create all possible nodes under input node, 
	cut the ones below threshold 
	load positions from cache
	'''
	board_id = make_id(node.board)
	if cache.get(board_id)==None:
		cache.put(board_id, pickle.load(open(os.path.join(preoprocessed_data_path, puzzle, board_id)+'.p', 'rb')))
	all_children = cache.get(board_id)
	for i in range(len(all_children['children_ids'])):
		child = Node(all_children['children_boards'][i], all_children['children_mags'][i], params, parent=node)
		node.children.append(child)
	Vmaxchild = ArgmaxChild(node)
	for child_idx in range(len(node.children))[::-1]: # iterate in reverse order
		if abs(node.children[child_idx].value - Vmaxchild.value) > params.pruning_threshold:
			node.remove_child(child_idx)
	
def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	this_node.value = ArgmaxChild(this_node).value
	if this_node != root_node:
		Backpropagate(this_node.parent, root_node)

def ArgmaxChild(node): 
	''' 
		return the child with max value 
	'''
	return max(node.children, key=attrgetter('value'))

def MakeMove(root, params, cache, puzzle, hit=False, verbose=False):
	''' 
	`	returns an optimal move to make next 
		according to value function and current board position
	'''

	assert len(root.children) == 0
	if hit:
		return root
	# check if root node is already a winning position
	if is_solved(root.board):
		# move red car to position 16 if rootis a winning position
		board = move(root.board, 'r', 16)
		result = Node(board, None, params)
		return result
	if Lapse(params.lapse_rate): # random move
		return RandomMove(root, params)
	else:
		DropFeatures(params.feature_dropping_rate)
		while not Stop(probability=params.stopping_probability):
			leaf, leaf_is_solution = SelectNode(root)
			if verbose:
				print('\n\nnew while')
				print('Leaf Node is solution: '+str(leaf_is_solution))
				print('leaf Node\n'+str(leaf.board_to_str()))
				print('\n\n')
			if leaf_is_solution:
				Backpropagate(leaf.parent, root)
				continue
			ExpandNode(leaf, params, cache, puzzle)
			if verbose:
				leaf.print_children()
			Backpropagate(leaf, root)
	if root.children == []: # if did not enter while loop at all
		ExpandNode(root, params, cache, puzzle)
	return ArgmaxChild(root)


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


def prepare_ibs(subject_file='A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9.csv',
				subject_path='/Users/yichen/Desktop/subjects/',
				preoprocessed_data_path='/Users/yichen/Desktop/preprocessed_positions/',
				instancepath='/Users/yichen/Documents/RushHour/exp_data/data_adopted/'): # sequantial
	'''
	preparation for ibs: 
	load some boards into cache, 
	calculate log likelihood for lower bound,
	return list of subject moves and answers
	'''
	puzzle_cache = {}
	# read subject data
	if subject_file == '':
		subject_file = random.choice(os.listdir(subject_path))
	print(subject_file)
	df = pd.read_csv(os.path.join(subject_path, subject_file))
	
	# initialize early stopping LL and hit array
	subject_data = [] # list of board ids
	subject_answer = [] # list of board ids as subject answer
	subject_puzzle = [] # list of puzzle
	LL_lower = 0 # lower bound
	children_count = [] # number of children for each move
	puzzle = '' # keep track of puzzle id at each data row

	# iterate through each move from data file
	for idx, row in df.iterrows():
		if row['event'] == 'start': # load new puzzle
			instance = row['instance']
			ins_file = instancepath+instance+'.json'
			board = json_to_board(ins_file)
			if puzzle != row['instance']:
				puzzle = row['instance']
				print('index '+str(idx))
				if puzzle_cache.get(puzzle)==None:
					cache=LRUCache(puzzle)
					puzzle_cache[puzzle]=cache
				else:
					cache=puzzle_cache[puzzle]
			continue
		if row['piece']=='r' and row['move']==16 and row['optlen']==1: # skip the winning moves
			continue
		# prepare root node
		root = Node(board, None, None)
		# append move data
		root_id = make_id(board)
		subject_data.append(root_id)
		subject_puzzle.append(puzzle)
		# load children and count children number
		children_pickle = pickle.load(open(os.path.join(preoprocessed_data_path, puzzle, root_id)+'.p', 'rb'))
		puzzle_cache[puzzle].put(root_id, children_pickle)
		children_count.append(len(children_pickle['children_ids']))
		# change the board and make the current move (answer)
		piece = str(row['piece'])
		move_to = int(row['move'])
		board = move(board, piece, move_to)
		# add answer to list
		subject_answer.append(make_id(board))
	
	# calculate current LL and LLlower
	LL_lower = np.sum([np.log(1.0/n) for n in children_count])
	print('subject data size '+str(len(subject_data)))
	print('subject answer size '+str(len(subject_answer)))
	print('subject puzzle size '+str(len(subject_puzzle)))
	print('mean children count '+str(np.mean(children_count)))
	print('LL_lower '+str(LL_lower))

	return puzzle_cache, LL_lower, subject_data, subject_answer, subject_puzzle



def ibs_early_stopping(inparams,  
					puzzle_cache,
					LL_lower,
					subject_data,
					subject_answer, 
					subject_puzzle,
					threshold_num=100): 
	'''
	implement ibs with early stopping
	sequential
	returns the log likelihood of current subject
	'''
	start_time = time.time()
	
	# initialize parameters
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
							w4=inparams[3], w5=inparams[4], w6=inparams[5], 
							w7=inparams[6], 
							stopping_probability=inparams[7],
							pruning_threshold=inparams[8],
							lapse_rate=inparams[9])
	
	# initialize iteration data
	hit_target = [False]*len(subject_data) # True if hit for each move
	count_iteration = [1]*len(subject_data) # count of iteration for each move
	k = 0
	LL_k = 0
	previous_hit = 0
	count_repeat = 0

	# iterate until meets early stopping criteria
	while hit_target.count(False) > 0:
		if LL_k	<= LL_lower:
			LL_k = LL_lower
			print('*********************** exceeds LL lower bound, break')
			break
		if count_repeat >= threshold_num:
			LL_k = LL_lower
			print('*********************** hit number stays same for '+str(threshold_num)+' iterations, break')
			break
		LL_k = 0
		k += 1
		# print('Iteration k='+str(k))
		print('puzzle_cache length for puzzle '+str(list(puzzle_cache.keys())[0])+': '+str(len(puzzle_cache[list(puzzle_cache.keys())[0]].cache)))
		for idx in range(len(subject_data)):
			if hit_target[idx]: # if current move was already hit by previous iterations
				LL_k += harmonic_sum(count_iteration[idx])
				continue
			decision = MakeMove(Node(id_to_board(subject_data[idx],subject_puzzle[idx]),None,params), params, puzzle_cache[subject_puzzle[idx]], subject_puzzle[idx])
			if make_id(decision.board)==subject_answer[idx]: # hit
				hit_target[idx] = True
				LL_k += harmonic_sum(count_iteration[idx])
			else: # not hit
				count_iteration[idx] += 1
		LL_k = -LL_k - (hit_target.count(False))*harmonic_sum(k)
		hit_number = hit_target.count(True)
		print('\tIteration k='+str(k)+', hit_target '+str(hit_number)+', previous_hit '+str(previous_hit)+', Kth LL_k='+str(LL_k))
		if hit_number == previous_hit:
			count_repeat += 1
			print('count repeat increased to '+str(count_repeat)+' for hit number '+str(hit_number))
		else:
			count_repeat = 0
		previous_hit = hit_number

	print('IBS total time lapse '+str(time.time() - start_time))
	print('Final LL_k: '+str(LL_k))
	return LL_k



# def ibs_early_stopping_parallel(inparams,  
# 					puzzle_cache,
# 					LL_lower,
# 					subject_data,
# 					subject_answer, 
# 					subject_puzzle,
# 					pool,
# 					threshold_num=80,
# 					func=MakeMove): 
# 	'''
# 	implement ibs with early stopping
# 	parallel computing
# 	returns the log likelihood of current subject
# 	'''
# 	start_time = time.time()
	
# 	# initialize parameters
# 	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
# 							w4=inparams[3], w5=inparams[4], w6=inparams[5], 
# 							w7=inparams[6], 
# 							stopping_probability=inparams[7],
# 							pruning_threshold=inparams[8],
# 							lapse_rate=inparams[9])
	
# 	# initialize iteration data
# 	hit_target = [False]*len(subject_data) # True if hit for each move
# 	count_iteration = [1]*len(subject_data) # count of iteration for each move
# 	k = 0
# 	LL_k = 0
# 	previous_hit = 0
# 	count_repeat = 0

# 	# iterate until meets early stopping criteria
# 	while hit_target.count(False) > 0:
# 		if LL_k	<= LL_lower:
# 			LL_k = LL_lower
# 			print('*********************** exceeds LL lower bound, break')
# 			break
# 		if count_repeat >= threshold_num:
# 			LL_k = LL_lower
# 			print('*********************** hit number stays same for '+str(threshold_num)+' iterations, break')
# 			break
# 		LL_k = 0
# 		k += 1
# 		print('Iteration k='+str(k))
# 		print('puzzle_cache length for puzzle '+str(list(puzzle_cache.keys())[0])+': '+str(len(puzzle_cache[list(puzzle_cache.keys())[0]].cache)))
# 		decisions =  [pool.apply_async(func, args=(Node(id_to_board(subid, subpuzzle),None,params), params, puzzle_cache[subpuzzle],subpuzzle,hit)).get() for subid,subpuzzle,hit in zip(subject_data,subject_puzzle,hit_target)]
# 		hit_target = [a or b for a,b in zip(hit_target, [make_id(decision.board)==answer for decision, answer in zip(decisions, subject_answer)])]
# 		for i in range(len(subject_data)):
# 			if not hit_target[i]:
# 				count_iteration[i] += 1
# 			else:
# 				LL_k += harmonic_sum(count_iteration[i])
# 		LL_k = -LL_k - (hit_target.count(False))*harmonic_sum(k)
# 		hit_number = hit_target.count(True)
# 		print('\thit_target '+str(hit_number)+', previous_hit '+str(previous_hit))
# 		print('\tKth LL_k '+str(LL_k))
# 		if hit_number == previous_hit:
# 			count_repeat += 1
# 			print('count repeat increased to '+str(count_repeat)+' for hit number '+str(hit_number))
# 		else:
# 			count_repeat = 0
# 		previous_hit = hit_number

# 	print('IBS total time lapse '+str(time.time() - start_time))
# 	print('Final LL_k: '+str(LL_k))
# 	return LL_k


# x0 = matlab.double([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
# 			0.01, 10, 0.01])
# lb = matlab.double([-5, -5, -5, -5, -5, -5, -5,
# 			0, 0, 0])
# ub = matlab.double([5, 5, 5, 5, 5, 5, 5,
# 			1, 50, 1])
# plb = matlab.double([-1, -1, -1, -1, -1, -1, -1,
# 			0, 1, 0])
# pub = matlab.double([5, 5, 5, 5, 5, 5, 5,
# 			0.5, 20, 0.5])

puzzle_cache, LL_lower, subject_data, subject_answer, subject_puzzle = prepare_ibs()

# eng = matlab.engine.start_matlab()

def ibs_interface(w1,w2,w3,w4,w5,w6,w7,sp,pt,lr):
	inparams = [w1,w2,w3,w4,w5,w6,w7,sp,pt,lr]
	# global eng, puzzle_cache, LL_lower, subject_data, subject_answer, subject_puzzle
	# inparams = np.asarray(eng.workspace['x0'])
	print('inparams '+str(inparams))
	return ibs_early_stopping(inparams,  
					puzzle_cache,
					LL_lower,
					subject_data,
					subject_answer, 
					subject_puzzle)


# if __name__ == '__main__':

	# x0 = matlab.double([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
	# 			0.01, 10, 0.01])
	# lb = matlab.double([-5, -5, -5, -5, -5, -5, -5,
	# 			0, 0, 0])
	# ub = matlab.double([5, 5, 5, 5, 5, 5, 5,
	# 			1, 50, 1])
	# plb = matlab.double([-1, -1, -1, -1, -1, -1, -1,
	# 			0, 1, 0])
	# pub = matlab.double([5, 5, 5, 5, 5, 5, 5,
	# 			0.5, 20, 0.5])

	# puzzle_cache, LL_lower, subject_data, subject_answer, subject_puzzle = prepare_ibs()


	# eng = matlab.engine.start_matlab()
	# sys.setrecursionlimit(31000)
	# result = eng.bads("@ll", x0,lb,ub,plb,pub, nargout=2)
	# print(result)

	# eng = matlab.engine.start_matlab()
	# x0 = matlab.double([0, 0]) #Starting point
	# lb = matlab.double([-20, -20]) # Lower bounds
	# ub = matlab.double([20,20]) #               % Upper bounds
	# plb = matlab.double([-5,-5]) #              % Plausible lower bounds
	# pub = matlab.double([5,5]) #             % Plausible upper bounds

	# result = eng.bads("@rosenbrocks", x0,lb,ub,plb,pub, nargout=2)
	# print(result)
	

	'''
	preload=0: no preload, load necessary positions in ExpandNode; 
	1: preload all positions at once; 
	2: preload one puzzle as needed;
	3: load positions as necessary and keep cache
	'''
	# puzzle_cache = {}
	# inparams = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
	# 			0.01, 10, 0.01]
	# puzzle_cache, LL_lower, subject_data, subject_answer, subject_puzzle = prepare_ibs(puzzle_cache)


	# # threads = mp.cpu_count()
	# # print('number of cpus '+str(threads))
	# # pool = mp.Pool(processes=threads)
	# ibs_early_stopping(inparams,  
	# 				puzzle_cache,
	# 				LL_lower,
	# 				subject_data,
	# 				subject_answer, 
	# 				subject_puzzle)
	



