'''
comparison between BFS and Myopic
python3
'''
import BFS, Myopic
from BFS import Node
import time
import multiprocessing as mp
import pandas as pd

def avg_ll(params, num_iteration, datapath='/Users/chloe/Desktop/A289D98Z4GAZ28-3ZV9H2YQQEFR2R3JK2IXZUJPXAF3WW.xlsx'): # parallel computing
	start_time = time.time()
	list_carlist, user_choice = BFS.load_data(datapath)
	total_ll_bfs = 0
	total_ll_myopic = 0
	for iteration in range(num_iteration):
		print('Iteration num '+str(iteration))
		pool = mp.Pool(processes=mp.cpu_count())
		hit_target_bfs = [False]*len(list_carlist)
		hit_target_myopic = [False]*len(list_carlist)
		count_iteration_bfs = [0]*len(list_carlist)
		count_iteration_myopic = [0]*len(list_carlist)
		print('Sample size '+str(len(list_carlist)))
		list_rootnode = [Node(cur_root, params) for cur_root in list_carlist]
		list_answer = [Node(cl, params).board_to_str() for cl in user_choice]
		print('Params sp '+str(params.stopping_probability))
		count_iteration_bfs = [x+1 for x in count_iteration_bfs]
		count_iteration_myopic = [x+1 for x in count_iteration_myopic]
		# start iteration
		k = 0
		LL_k = 0
		while hit_target_myopic.count(False) > 0:
			LL_k_bfs = 0
			LL_k_myopic = 0
			k += 1
			print('Iteration K='+str(k))
			list_rootnode = [Node(cur_root, params) for cur_root in list_carlist]
			model_decision_bfs = [pool.apply_async(BFS.MakeMove, args=(cur_root, params, hit)).get() for cur_root, hit in zip(list_rootnode, hit_target_bfs)]
			model_decision_myopic = [pool.apply_async(Myopic.MakeMove, args=(cur_root, params, hit)).get() for cur_root, hit in zip(list_rootnode, hit_target_myopic)]
			for i in range(len(count_iteration_bfs)):
				if not hit_target_bfs:
					count_iteration_bfs[i] += 1
				if not hit_target_myopic:
					count_iteration_myopic[i] += 1
			hit_target_bfs = [a or b for a,b in zip(hit_target_bfs, [decision.board_to_str()==answer for decision, answer in zip(model_decision_bfs, list_answer)])]
			hit_target_myopic = [a or b for a,b in zip(hit_target_myopic, [decision.board_to_str()==answer for decision, answer in zip(model_decision_myopic, list_answer)])]
			for i in range(len(count_iteration_bfs)):
				if hit_target_bfs:
					LL_k_bfs += BFS.harmonic_sum(count_iteration_bfs[i])
				if hit_target_myopic:
					LL_k_myopic += BFS.harmonic_sum(count_iteration_myopic[i])
			LL_k_bfs = (-1.0/len(hit_target_bfs))*LL_k_bfs - (hit_target_bfs.count(False)/len(hit_target_bfs))*BFS.harmonic_sum(k)
			LL_k_myopic = (-1.0/len(hit_target_myopic))*LL_k_myopic - (hit_target_myopic.count(False)/len(hit_target_myopic))*BFS.harmonic_sum(k)
			print('\thit_target_bfs '+str(hit_target_bfs.count(True)))
			print('\thit_target_myopic '+str(hit_target_myopic.count(True)))
			print('\tKth LL_k_bfs '+str(LL_k_bfs))
			print('\tKth LL_k_myopic '+str(LL_k_myopic))
		pool.close()
		pool.join()
		print('IBS total time lapse '+str(time.time() - start_time))
		print('Final LL_k_bfs: '+str(LL_k_bfs))
		print('Final LL_k_myopic: '+str(LL_k_myopic))
		total_ll_bfs += LL_k_bfs
		total_ll_myopic += LL_k_myopic
	return total_ll_bfs/num_iteration, total_ll_myopic/num_iteration

if __name__ == '__main__':
	params = BFS.Params(0.7,0.6,0.5,0.4,0.3,0.2,0.1, 
					stopping_probability=0.1,
					feature_dropping_rate=0.0, 
					pruning_threshold=10.0, 
					lapse_rate=0.05,
					mu=0.0, sigma=1.0)
	avg_ll(params, 3)


