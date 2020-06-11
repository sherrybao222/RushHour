''' 
check if subject data has mismatch answer
make sure that subject answer is one of the children from subject board position
python3
'''

from BFS_ibs import *
import os

preoprocessed_data_path = '/Users/yichen/Desktop/preprocessed_positions/'
all_subject_files = os.listdir('/Users/yichen/Desktop/subjects/')
# subject_file = 'A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9.csv'
for subject_file in all_subject_files:
	print('------ checking subject '+subject_file)
	puzzle_cache = {}
	inparams = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 0.01]
	params = Params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
					w4=inparams[3], w5=inparams[4], w6=inparams[5], 
					w7=inparams[6], 
					stopping_probability=inparams[7],
					pruning_threshold=inparams[8],
					lapse_rate=inparams[9])

	puzzle_cache, LL_lower, subject_data, subject_answer, subject_puzzle = prepare_ibs(puzzle_cache, subject_file=subject_file)

	for idx in range(len(subject_data)):
		# print('----- checking index '+str(idx))
		#check if original children contain subject move
		identity = subject_data[idx]
		org_root = puzzle_cache[subject_puzzle[idx]].get(identity)
		if org_root==None:
			puzzle_cache[subject_puzzle[idx]].put(identity, pickle.load(open(os.path.join(preoprocessed_data_path, subject_puzzle[idx], identity)+'.p', 'rb')))
			org_root = puzzle_cache[subject_puzzle[idx]].get(identity)
		org_children = org_root['children_ids']
		found1 = False
		for child in org_children:
			if child == subject_answer[idx]:
				found1 = True
				break
		if found1==False:
			print('Answer not found at original data, idx='+str(idx)+', root '+subject_data[idx])
			print('\tChildren: '+str([identity for identity in org_children]))
			prit('\tAnswer child: '+str(subject_answer[idx]))

		# check if processed children contain subject move
		newroot = MakeMove(Node(id_to_board(subject_data[idx],subject_puzzle[idx]),None,params), params, puzzle_cache[subject_puzzle[idx]], subject_puzzle[idx])
		found2 = False
		for child in newroot.children:
			if make_id(child.board)==subject_answer[idx]:
				found2 = True
				break
		if found2==False:
			print('Answer not found after MakeMove, idx='+str(idx)+', root '+subject_data[idx])
			print('\tChildren: '+str([make_id(child.board) for child in newroot.children]))
			print('\tOriginal children: '+str([identity for identity in puzzle_cache[subject_puzzle[idx]].get(subject_data[idx])['children_ids']]))
			print('\tAnswer child: '+str(subject_answer[idx]))