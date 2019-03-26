# post-process moves_sol.npy
# need to run with 
import numpy as np

file_dir = '/Users/chloe/Documents/RushHour/exp_data/moves_sol1000.npy'
out_dir = '/Users/chloe/Documents/RushHour/exp_data/moves_sol1000_final.npy'
data = np.load(file_dir)
out_data = []

for i in range(0, len(data)):
	out_data.append(data[i])
	if data[i] == []:
		out_data.append([])

np.save(out_dir, out_data)
