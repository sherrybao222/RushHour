# PLOTTING: characterize trialdata each move
# probability matrix of error as a function of difflen(suboptimality score) and mobility
# saturation (alpha) coded by sample size in each entry
# plot mobility x diffoptlen trajectories of sample trials
# need to run with python27
import sys, csv, math
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy.stats as st
from matplotlib.colors import ListedColormap
np.seterr(divide='ignore', invalid='ignore')

all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
ins_optlen = [7] * len(all_instances)
ins_optlen[18:36]=[11]*len(ins_optlen[18:36])
ins_optlen[36:53]=[14]*len(ins_optlen[36:53])
ins_optlen[53:]=[16]*len(ins_optlen[53:])

optlen_consider = 7

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'

out_file1 = '/Users/chloe/Desktop/diffoptlen_mob_to_error_matrix'+str(optlen_consider)+'.png'
out_file2 = '/Users/chloe/Desktop/diffoptlen_mob_to_error_colorbar.png'
out_file3 = '/Users/chloe/Desktop/diffoptlen_mob_to_error_matrix'+str(optlen_consider)+'_prev_error.png'
out_file4 = '/Users/chloe/Desktop/diffoptlen_mob_to_error_matrix'+str(optlen_consider)+'_prev_nonerror.png'
out_file5 = '/Users/chloe/Desktop/traj_level_'+str(optlen_consider)+'_2.png'

move_data = pd.read_csv(moves_file)

error_diffoptlen_mob = []
count_diffoptlen_mob = []

error_diffoptlen_mob_preverror = [] # conditioned on that previous move was error
count_diffoptlen_mob_preverror = []

error_diffoptlen_mob_prevnonerror = [] # conditioned on that previous move was nonerror
count_diffoptlen_mob_prevnonerror = []

max_diffoptlen = ''
min_diffoptlen = ''
range_diffoptlen = ''
max_mob = ''

prev_error = ''

random_mod = 90 # for plotting trajectory
traj_flag = False
all_traj_diffoptlen = []
all_traj_mob = []
cur_traj_diffoptlen = []
cur_traj_mob = []



#################################### PROCESS DATA ###############################

for i in range(len(move_data)):
	
	row = move_data.loc[i, :]
	# first line
	if i == 0: 
		range_diffoptlen = int(row['range_diffoptlen'])
		min_diffoptlen = int(row['min_diffoptlen'])
		max_diffoptlen = int(row['max_diffoptlen'])
		max_mob = int(row['max_mobility'])
		error_diffoptlen_mob = np.zeros((range_diffoptlen+1, max_mob+1))
		error_diffoptlen_mob_preverror = np.zeros((range_diffoptlen+1, max_mob+1))
		error_diffoptlen_mob_prevnonerror = np.zeros((range_diffoptlen+1, max_mob+1))
		count_diffoptlen_mob = np.zeros((range_diffoptlen+1, max_mob+1))
		count_diffoptlen_mob_preverror = np.zeros((range_diffoptlen+1, max_mob+1))
		count_diffoptlen_mob_prevnonerror = np.zeros((range_diffoptlen+1, max_mob+1))
	
	# only consider puzzles in one level
	cur_ins = row['instance']
	if ins_optlen[all_instances.index(cur_ins)] != optlen_consider:
		continue 

	# clean prev_error if new trial starts
	event = row['event']
	if event == 'start':
		prev_error = 0
		# start to collect coordinate when random mod satisfies
		if i % random_mod == 0:
			traj_flag = True
			cur_traj_mob = []
			cur_traj_diffoptlen = []
		else:
			traj_flag = False # finish this trial
			all_traj_mob.append(cur_traj_mob)
			all_traj_diffoptlen.append(cur_traj_diffoptlen)

	# read data
	diffoptlen = row['diffoptlen']
	error = row['error_tomake']
	mobility = row['mobility']

	# save trajectory coordinate
	if traj_flag:
		cur_traj_diffoptlen.append(diffoptlen)
		cur_traj_mob.append(mobility)

	# hash total count
	count_diffoptlen_mob[abs(min_diffoptlen) + diffoptlen, mobility] += 1
	if prev_error == 1:
		count_diffoptlen_mob_preverror[abs(min_diffoptlen) + diffoptlen, mobility] += 1
	else:
		count_diffoptlen_mob_prevnonerror[abs(min_diffoptlen) + diffoptlen, mobility] += 1
	
	# hash error count
	if error == 1: 
		error_diffoptlen_mob[abs(min_diffoptlen) + diffoptlen, mobility] += 1
		if prev_error == 1:
			error_diffoptlen_mob_preverror[abs(min_diffoptlen) + diffoptlen, mobility] += 1
		else:
			error_diffoptlen_mob_prevnonerror[abs(min_diffoptlen) + diffoptlen, mobility] += 1

	# update
	prev_error = error


# compute probability
prob_error_diffoptlen_mob = error_diffoptlen_mob / count_diffoptlen_mob
prob_error_diffoptlen_mob_preverror = error_diffoptlen_mob_preverror / count_diffoptlen_mob_preverror
prob_error_diffoptlen_mob_prevnonerror = error_diffoptlen_mob_prevnonerror / count_diffoptlen_mob_prevnonerror

####################################### PLOTTING ##################################


# plot trajectory coordinate
for i in range(0, len(all_traj_mob)):
	plt.plot(all_traj_mob[i], all_traj_diffoptlen[i], '-p', linewidth=1, markersize=2)
plt.xlabel('Mobility')
plt.ylabel('Diffoptlen')
plt.title('Trajectory, Level '+str(optlen_consider))
plt.savefig(out_file5)
sys.exit()







# condition on previous move is error
fig, ax = plt.subplots(figsize=(5,6))
cm = plt.cm.get_cmap('cool')
my_cmap = cm(prob_error_diffoptlen_mob_preverror)

# normalize with count to alphas
alphas = np.log(count_diffoptlen_mob_preverror) / np.log(count_diffoptlen_mob_preverror.max())
my_cmap[:, :, -1] = alphas

ax.imshow(my_cmap)
cmap=plt.cm.get_cmap('cool')
norm = mp.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(0,1,11))
if optlen_consider != 16:
	ax.axhline(-min_diffoptlen - optlen_consider, \
			color='gray', linestyle='dashed', linewidth=1.5)
plt.yticks(np.arange(0,range_diffoptlen,4), np.arange(min_diffoptlen,max_diffoptlen,4))
plt.xticks(np.arange(0,max_mob,4), np.arange(0,max_mob,4))
plt.xlabel('Mobility')
plt.ylabel('Relative Distance to Goal')
plt.title('Prob of Error (Error in Previous Move), Level '+str(optlen_consider))
plt.savefig(out_file3)
plt.close()



# condition on previous move is nonerror
fig, ax = plt.subplots(figsize=(5,6))
cm = plt.cm.get_cmap('cool')
my_cmap = cm(prob_error_diffoptlen_mob_prevnonerror)

# normalize with count to alphas
alphas = np.log(count_diffoptlen_mob_prevnonerror) / np.log(count_diffoptlen_mob_prevnonerror.max())
my_cmap[:, :, -1] = alphas

ax.imshow(my_cmap)
cmap=plt.cm.get_cmap('cool')
norm = mp.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(0,1,11))
if optlen_consider != 16:
	ax.axhline(-min_diffoptlen - optlen_consider, \
			color='gray', linestyle='dashed', linewidth=1.5)
plt.yticks(np.arange(0,range_diffoptlen,4), np.arange(min_diffoptlen,max_diffoptlen,4))
plt.xticks(np.arange(0,max_mob,4), np.arange(0,max_mob,4))
plt.xlabel('Mobility')
plt.ylabel('Relative Distance to Goal')
plt.title('Prob of Error (Non-error in Previous Move), Level '+str(optlen_consider))
plt.savefig(out_file4)
plt.close()





fig, ax = plt.subplots(figsize=(5,6))
cm = plt.cm.get_cmap('cool')
my_cmap = cm(prob_error_diffoptlen_mob)

# normalize with count to alphas
alphas = np.log(count_diffoptlen_mob) / np.log(count_diffoptlen_mob.max())
my_cmap[:, :, -1] = alphas

ax.imshow(my_cmap)
cmap=plt.cm.get_cmap('cool')
norm = mp.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=np.linspace(0,1,11))
if optlen_consider != 16:
	ax.axhline(-min_diffoptlen - optlen_consider, \
			color='gray', linestyle='dashed', linewidth=1.5)
plt.yticks(np.arange(0,range_diffoptlen,4), np.arange(min_diffoptlen,max_diffoptlen,4))
plt.xticks(np.arange(0,max_mob,4), np.arange(0,max_mob,4))
plt.xlabel('Mobility')
plt.ylabel('Relative Distance to Goal')
plt.title('Prob of Error, Level '+str(optlen_consider))
plt.savefig(out_file1)
plt.close()






fig, ax = plt.subplots()
cm = plt.cm.get_cmap('cool')
fakedata = np.tile(np.linspace(0,1,50), (50,1))
my_cmap = cm(fakedata)
fakealphas = np.tile(np.linspace(0,1,50),(50,1)).T
my_cmap[:,:,-1] = fakealphas
ax.imshow(my_cmap)
plt.yticks([0,10,20,30,40,49], [0,0.2,0.4,0.6,0.8,1.0])
plt.xticks([0,10,20,30,40,49], [0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel('Prob of Error')
plt.ylabel('log(count) / log(count_max)')
plt.title('2D Colorbar')
plt.savefig(out_file2)
plt.close()




sys.exit()







