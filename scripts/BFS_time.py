''' 
measure time comlexity
BFS model self-defined ll function,
speeded version, prepared for BADS in MATLAB,
python3 or py27
'''
import random, copy, pickle, os, sys, time
from operator import attrgetter
import multiprocessing as mp
import numpy as np
from numpy import recfromcsv
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
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
	return random.choice(node.children)
	
def InitializeChildren(node, params):
	''' 
		initialize the list of children nodes
		(using all legal moves) 
	'''
	InitializeChildren_start = time.time()
	
	# find all legal moves
	all_moves = all_legal_moves(node.car_list, node.board)
	# print('len(all_legal_moves)='+str(len(all_moves)))
	time_dict['all_legal_moves'].append(time.time() - InitializeChildren_start)
	
	for (tag, pos) in all_moves:

		move_xy_start = time.time()
		# move board position to simulate possible move 
		new_list, _ = move_xy(node.car_list, tag, pos[0], pos[1])
		time_dict['move_xy'].append(time.time() - move_xy_start)

		childNode_start = time.time()	
		# create child node
		child = Node(new_list, params)
		child.parent = node
		node.children.append(child)
		time_dict['create_Node'].append(time.time() - childNode_start)
		time_dict['Node_board_time'].append(child.board_time)
		time_dict['Node_construct_mag_time'].append(child.construct_mag_time)
		time_dict['Node_assign_level_time'].append(child.assign_level_time)
		time_dict['Node_value_time'].append(child.value_time)
	
	# print('len(children) in InitializeChildren='+str(len(node.children)))
	time_dict['InitializeChildren'].append(time.time() - InitializeChildren_start)

def SelectNode(root_node):
	''' return the child with max value '''
	n = root_node
	while len(n.children) != 0:
		n = ArgmaxChild(n)
	# print('[[[ SelectNode start:\nselected node\n'+str(n.board_to_str()))
	# print('root node:\n'+str(root_node.board_to_str())+'\n end SelectNode ]]]')
	return n, is_solved(n.board, n.red)
 
def ExpandNode(node, params):
	''' 
	create all possible nodes under input node, 
	cut the ones below threshold 
	'''
	# print('[[[ ExpandNode start')
	InitializeChildren(node, params)
	Vmaxchild = ArgmaxChild(node)
	for child_idx in range(len(node.children))[::-1]: # iterate in reverse order
		if abs(node.children[child_idx].value - Vmaxchild.value) > params.pruning_threshold:
			node.remove_child(child_idx)
	# print('len(node.children)in ExpandNode='+str(len(node.children)))
	# print('end ExpandNode ]]]')

def Backpropagate(this_node, root_node):
	''' update value back until root node '''
	this_node.value = ArgmaxChild(this_node).value
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
	

def MakeMove(root, params, hit=False, verbose=False):
	''' 
	`	returns an optimal move to make next 
		according to value function and current board position
	'''
	MakeMove_start = time.time()
	beforewhile_start = time.time()
	# print('[[[[[[[[[[ MakeMove start')
	if hit: # for ibs, if matches human decision, return root node
		time_dict['MakeMove'].append(time.time()- MakeMove_start)
		return root
	assert len(root.children) == 0
	# check if root node is already a winning position
	if is_solved(root.board, root.red):
		# move red car to position 16 if rootis a winning position
		new_carlist, _ = move(root.car_list, 'r', 16)
		result = Node(new_carlist, params)
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
			ExpandNode(leaf, params)
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
		ExpandNode(root, params)
		time_dict['ExpandNode'].append(time.time() - ExpandNode_start)
	result = ArgmaxChild(root)
	time_dict['afterwhile'].append(time.time() - afterwhile_start)
	time_dict['MakeMove'].append(time.time()- MakeMove_start)
	# print('end MakeMove ]]]]]]]')
	return result

def test_makemove(datapath='/Users/chloe/Desktop/subjects/'):
	'''
	something is wrong when makemove processes this data
	'''
	# all_subjects = os.listdir(datapath)
	all_subjects=[
	# 'A1ZTSCPETU3UJW:3XC1O3LBOTUGQEPEV3HM8W67X4RTLL.csv', 
	# 'A2RCYLKY072XXO:3TAYZSBPLMG9ASQRWXURJVBCPLN2SH.csv', 
	# 'A1GQS6USF2JEYG:33F859I567LE8WC74WB3GA7E94XBHW.csv', 
	# 'A3Q0XAGQ7TD7MP:3GLB5JMZFY3TNXFGYMKRQ0JDXNTDGZ.csv', 
	# 'AXKM02NVXNGOM:33PPO7FECWN7JOLBOAKUBCWTC0AIDL.csv', 
	# 'A30AGR5KF8IEL:3EWIJTFFVPF14ZIVGF68BQEIRXP0ER.csv', 
	# 'ARWF605I7RWM7:3AZHRG4CU5SYU12YRVPCSZALW5003R.csv', 
	# 'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB.csv', 
	# 'A2IBQA01NR5N76:3X4MXAO0BHWJLTOLVSJTHSM54OHWRG.csv', 
	# 'A28RX7L0QZ993M:3SITXWYCNWHBUMCM90TPJWV8Y7AXBW.csv', 
	# 'A15781PHGW377Y:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I.csv', 
	# 'A53S7J4JGWG38:3M0BCWMB8W4W5M7WZVX3HDH1P53WB1.csv', 
	# 'A3GOOI75XOF24V:3TMFV4NEP9MD3O9PWJDTQBR0HZYW8I.csv', 
	# 'A15FXHC1CVNW31:3TS1AR6UQRM7SOIBWPBN8N9589Y7FV.csv', 
	# 'A1XDMS0KFSF5JW:3M0BCWMB8W4W5M7WZVX3HDH1M3FBWL.csv', 
	# 'A21S7MA9DCW95E:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK.csv', 
	# 'A3OB6REG5CGHK1:3STRJBFXOXZ5687WA35LTWTS7QZKTT.csv', 
	# 'A2RTJ6BDY0DZVZ:3IJXV6UZ1YR1KY4G6BFEG1DXP7SRI1.csv', 
	# 'A3NMQ3019X6YE0:3V5Q80FXIYZ5QB5C6ITQBN30WY823U.csv', 
	'A289D98Z4GAZ28:3ZV9H2YQQEFR2R3JK2IXZUJPXAF3WW.csv', 
	# 'ALH1K6ZAQQMN7:3IXEICO793RY7TM78ZBKJDOA7ADT6Q.csv', 
	# 'A1EY7WONSYGBVY:3O6CYIULEE9B1LG2ZMEYM39PDJKUWB.csv', 
	# 'A2QIZ31TMHU0GD:3FUI0JHJPY6UBT1VAI7VUX8S3KH33W.csv', 
	# 'A191V7PT3DQKDP:3PMBY0YE28B43VMUKKJ6EDF85RV9C8.csv', 
	# 'A18QU0YQB6Q8DF:3VE8AYVF8N5BS2NU6U3TMN50JMNF8V.csv', 
	'AO1QGAUZ85T6L:3HWRJOOET6A15827PHPSLWK1MOYSEU.csv', 
	# 'A23XJ8I86R0O3B:30IQTZXKALEAAZ9CBKW0ZFZP6O5X07.csv', 
	# 'A3GXC3VG37CQ3G:3TPWUS5F8A9FFRZ2DVTYSXNJ6BWWC6.csv', 
	# 'A3NMQ3019X6YE0:3I2PTA7R3U2SESF4TZBQORI5LSJQKK.csv', 
	# 'A13BZCNJ0WR1T7:3TYCR1GOTDRCCQYD1V64UK7OHTBZLQ.csv', 
	# 'AMW2XLD9443OH:37QW5D2ZRHUKW7SGCE3STMOFBMK8SX.csv', 
	# 'A214HWAW1PYWO8:34BBWHLWHBJ6SUL255PK30LEGY5IWG.csv', 
	# 'A18T3WK7J16C1B:3IJXV6UZ1YR1KY4G6BFEG1DXR47RIC.csv', 
	# 'A1N1EF0MIRSEZZ:3R5F3LQFV3SKIB1AENMWM1BICT5OZB.csv', 
	# 'A1LR0VQIHQUJAM:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I.csv', 
	'A30KYQGABO7JER:37TD41K0AIHM8AITTQJXV8KY05QCSB.csv', 
	'A3CTXNQ2GXIQSP:34HJIJKLP64Z5YMIU6IKNXSH7PDV4I.csv', 
	'A23437BMZ5T1FH:3IHR8NYAM89M0EPM8U9LH53ZJDTP4U.csv', 
	'A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9.csv', 
	'A3BPRPN10HJD4B:3TXD01ZLD5PZSJXIPG8FRBQYTI9U4X.csv', 
	'A1USR9JCAMDGM3:3PB5A5BD0WED6OE679H5Q89HBDGG71.csv', 
	'A1F4N58CAX8IMK:35DR22AR5ES6RR89U7EJ1DXW91AX3P.csv', 
	'A2MYB6MLQW0IGN:3TYCR1GOTDRCCQYD1V64UK7OE48LZS.csv']
	# all_subjects=['A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB.csv']
	instancepath='/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	# preset parameters
	inparams = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 0.01]
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
					w4=inparams[3], w5=inparams[4], w6=inparams[5], 
					w7=inparams[6], 
					stopping_probability=inparams[7],
					pruning_threshold=inparams[8],
					lapse_rate=inparams[9])
	# read in data
	for subject in all_subjects:
		subpath = datapath+subject
		df = pd.read_csv(subpath)
		# for idx, row in df.iloc[484:501].iterrows():
		for idx, row in df.iterrows():
			print('-------------- ord: '+str(row['ord'])+', subject '+str(subject))
			current_ord = row['ord']
			if row['event'] == 'start':
				instance = row['instance']
				cur_carlist = json_to_car_list(instancepath+instance+'.json')
				continue
			if row['piece']=='r' and row['move']==16 and row['optlen']==1:
				continue

			# call makemove and record time data
			MakeMove(Node(cur_carlist, params), params)
			
			# change the board and perform the current move
			piece = str(row['piece'])
			move_to = int(row['move'])
			cur_carlist, _ = move(cur_carlist, piece, move_to)



if __name__ == '__main__':
	# data path definitions
	all_datapath = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'
	outputpath = '/Users/chloe/Desktop/timedict_new.csv'
	instancepath='/Users/chloe/Documents/RushHour/exp_data/data_adopted/'
	subject_path = '/Users/chloe/Desktop/subjects/' + random.choice(os.listdir('/Users/chloe/Desktop/subjects/'))
	df = pd.read_csv(subject_path)
	print('subject '+str(subject_path)+'\nsample size: '+str(df.shape[0]))

	# initialize parameters
	inparams = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 0.01]
	all_time_dict = {'MakeMove':[], 
					'insidewhile':[], 'beforewhile':[], 'afterwhile':[],
					'Stop':[], 'SelectNode':[], 'Backpropagate':[], 'ArgmaxChild':[],
					'ExpandNode':[], 
						'InitializeChildren':[], 
							'all_legal_moves':[], 'move_xy':[], 
							'create_Node': [], 
								'Node_board_time':[], 'Node_construct_mag_time':[], 'Node_assign_level_time':[], 'Node_value_time':[],								
					}
        # for testing makemove on specific subjects only
	problem_ord = 1411
	verbose = False
	time_dict = all_time_dict
	test_makemove()
	sys.exit()
        # end, for testing makemove on specific subject only
	
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
					w4=inparams[3], w5=inparams[4], w6=inparams[5], 
					w7=inparams[6], 
					stopping_probability=inparams[7],
					pruning_threshold=inparams[8],
					lapse_rate=inparams[9])
	
	# iterate over each move
	for idx, row in df.iterrows():
		print(idx)
		print(subject_path)
		# initialize record for current move
		time_dict = {}
		for key in all_time_dict.keys():
			time_dict[key] = []
		# read current move data 
		if row['event'] == 'start':
			# load new instance data
			instance = row['instance']
			ins_file = instancepath+instance+'.json'
			cur_carlist = json_to_car_list(ins_file)
			continue
		if row['piece']=='r' and row['move']==16 and row['optlen']==1: # win
			# skip the winning moves
			continue

		# call makemove and record time data
		MakeMove(Node(cur_carlist, params), params)
		
		# change the board and perform the current move
		piece = str(row['piece'])
		move_to = int(row['move'])
		cur_carlist, _ = move(cur_carlist, piece, move_to)
		
		# summarize all records from current move
		for key in all_time_dict.keys():
			all_time_dict[key].append(sum(time_dict[key]))

	# all moves completed, convert to dataframe
	all_time_df = pd.DataFrame.from_dict(all_time_dict)
	# save files as csv and calculate summary statistics
	all_time_df.to_csv(outputpath, index = False, header=True)
	df = pd.read_csv(outputpath)
	for col in df.columns[1:]:
		df[col] = df[col]/df['MakeMove']
	print(df.mean())
















