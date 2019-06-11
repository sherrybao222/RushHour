# PLOTTING: characterize move data
# probability of making error as a function of 
# mobinc, tie_mobred, mobred
# contains 3 types of error:
# nonerror (correct move), mild_error (tie), severe_error (distance increase)
# need to run with python27
import sys, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
np.seterr(divide='ignore', invalid='ignore')

moves_file = '/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv'

out_file1 = '/Users/chloe/Desktop/prob_errors1.png' # normalize by column
out_file2 = '/Users/chloe/Desktop/prob_errors2.png' # normalize by row
out_file3 = '/Users/chloe/Desktop/prob_errors3.png' # normalize by all

move_data = pd.read_csv(moves_file)

'''
		MobInc 		TieMobred 		Mobred
NonError
MildError
SevereError
'''
prob = np.zeros((3,3))

# normalize by column
nonerror_mobinc = []
nonerror_tiemobred = []
nonerror_mobred = []

milderror_mobinc = []
milderror_tiemobred = []
milderror_mobred = []

severeerror_mobinc = []
severeerror_tiemobred = []
severeerror_nonmobred = []

count_mobinc = []
count_tiemobred = []
count_mobred = []

# normalize by row
mobinc_nonerror = []
mobinc_milderror = []
mobinc_severeerror = []

tiemobred_nonerror = []
tiemobred_milderror = []
tiemobred_severeerror = []

mobred_nonerror = []
mobred_milderror = []
mobred_severeerror = []

count_nonerror = []
count_milderror = []
count_severeerror = []


#################################### PROCESS DATA ###############################

for i in range(len(move_data)-1):
	row = move_data.loc[i, :]
	# first line
	if i == 0: 
		
		nonerror_mobinc = np.zeros(2)
		nonerror_tiemobred = np.zeros(2)
		nonerror_mobred = np.zeros(2)
	
		milderror_mobinc = np.zeros(2)
		milderror_tiemobred = np.zeros(2)
		milderror_mobred = np.zeros(2)

		severeerror_mobinc = np.zeros(2)
		severeerror_tiemobred = np.zeros(2)
		severeerror_mobred = np.zeros(2)
		
		count_mobinc = np.zeros(2)
		count_tiemobred = np.zeros(2)
		count_mobred = np.zeros(2)

		mobinc_nonerror = np.zeros(2)
		mobinc_milderror = np.zeros(2)
		mobinc_severeerror = np.zeros(2)

		tiemobred_nonerror = np.zeros(2)
		tiemobred_milderror = np.zeros(2)
		tiemobred_severeerror = np.zeros(2)

		mobred_nonerror = np.zeros(2)
		mobred_milderror = np.zeros(2)
		mobred_severeerror = np.zeros(2)

		count_nonerror = np.zeros(2)
		count_milderror = np.zeros(2)
		count_severeerror = np.zeros(2)
		
		


	nonerror = int(row['nonerror_tomake'])
	milderror = int(row['mild_error_tomake'])
	severeerror = int(row['severe_error_tomake'])

	count_nonerror[nonerror] += 1
	count_milderror[milderror] += 1
	count_severeerror[severeerror] += 1

	mobinc = int(row['mobinc'])
	tiemobred = int(row['tie_mobred'])
	mobred = int(row['mobility_reduced'])
	
	count_mobinc[mobinc] += 1
	count_tiemobred[tiemobred] += 1
	count_mobred[mobred] += 1

	# normalize by column
	if nonerror == 1:
		nonerror_mobinc[mobinc] += 1
		nonerror_tiemobred[tiemobred] += 1
		nonerror_mobred[mobred] += 1
	elif milderror == 1:
		milderror_mobinc[mobinc] += 1
		milderror_tiemobred[tiemobred] += 1
		milderror_mobred[mobred] += 1
	elif severeerror != 1:
		print('line '+str(i)+' severeerror=0')
	else: #severe error
		severeerror_mobinc[mobinc] += 1
		severeerror_tiemobred[tiemobred] += 1
		severeerror_mobred[mobred] += 1

	# normalize by row
	if mobinc == 1:
		mobinc_nonerror[nonerror] += 1
		mobinc_milderror[milderror] += 1
		mobinc_severeerror[severeerror] += 1
	elif tiemobred == 1:
		tiemobred_nonerror[nonerror] += 1
		tiemobred_milderror[milderror] += 1
		tiemobred_severeerror[severeerror] += 1
	else:
		mobred_nonerror[nonerror] += 1
		mobred_milderror[milderror] += 1
		mobred_severeerror[severeerror] += 1



################################ CALCULATE PROB AND PLOT #############################
# normalize by column
prob_nonerror_mobinc = nonerror_mobinc / count_mobinc
prob_nonerror_tiemobred = nonerror_tiemobred / count_tiemobred
prob_nonerror_mobred = nonerror_mobred / count_mobred
prob[0,0]=prob_nonerror_mobinc[1]
prob[0,1]=prob_nonerror_tiemobred[1]
prob[0,2]=prob_nonerror_mobred[1]

prob_milderror_mobinc = milderror_mobinc / count_mobinc
prob_milderror_tiemobred = milderror_tiemobred / count_tiemobred
prob_milderror_mobred = milderror_mobred / count_mobred
prob[1,0]=prob_milderror_mobinc[1]
prob[1,1]=prob_milderror_tiemobred[1]
prob[1,2]=prob_milderror_mobred[1]

prob_severeerror_mobinc = severeerror_mobinc / count_mobinc
prob_severeerror_tiemobred = severeerror_tiemobred / count_tiemobred
prob_severeerror_mobred = severeerror_mobred / count_mobred
prob[2,0]=prob_severeerror_mobinc[1]
prob[2,1]=prob_severeerror_tiemobred[1]
prob[2,2]=prob_severeerror_mobred[1]

print(prob)
print(np.sum(prob, axis=0))
print(np.sum(prob, axis=1))

f, ax = plt.subplots(figsize=(6,5))
sns.heatmap(prob, annot=True, fmt="f", linewidths=.5, ax=ax, cmap=sns.color_palette("Blues"))
plt.xticks([0.5, 1.5, 2.5], ['Mob+', 'Mob0', 'Mob-'])
plt.yticks([0.5, 1.5, 2.5], ['NonError', 'MildError', 'SevereError'])
plt.suptitle('Probability of Different Errors')
plt.title('Normalized by Column')
plt.savefig(out_file1)
plt.close()



# normalize by row
prob_nonerror_mobinc = mobinc_nonerror / count_nonerror
prob_nonerror_tiemobred = tiemobred_nonerror / count_nonerror
prob_nonerror_mobred = mobred_nonerror / count_nonerror
prob[0,0]=prob_nonerror_mobinc[1]
prob[0,1]=prob_nonerror_tiemobred[1]
prob[0,2]=prob_nonerror_mobred[1]

prob_milderror_mobinc = mobinc_milderror / count_milderror
prob_milderror_tiemobred = tiemobred_milderror / count_milderror
prob_milderror_mobred = mobred_milderror / count_milderror
prob[1,0]=prob_milderror_mobinc[1]
prob[1,1]=prob_milderror_tiemobred[1]
prob[1,2]=prob_milderror_mobred[1]

prob_severeerror_mobinc = mobinc_severeerror / count_severeerror
prob_severeerror_tiemobred = tiemobred_severeerror / count_severeerror
prob_severeerror_mobred = mobred_severeerror / count_severeerror
prob[2,0]=prob_severeerror_mobinc[1]
prob[2,1]=prob_severeerror_tiemobred[1]
prob[2,2]=prob_severeerror_mobred[1]

print(prob)
print(np.sum(prob, axis=0))
print(np.sum(prob, axis=1))

f, ax = plt.subplots(figsize=(6,5))
sns.heatmap(prob, annot=True, fmt="f", linewidths=.5, ax=ax, cmap=sns.color_palette("Blues"))
plt.xticks([0.5, 1.5, 2.5], ['Mob+', 'Mob0', 'Mob-'])
plt.yticks([0.5, 1.5, 2.5], ['NonError', 'MildError', 'SevereError'])
plt.suptitle('Probability of Different Errors')
plt.title('Normalized by Row')
plt.savefig(out_file2)
plt.close()



# normalize by all (entire grid)
countall = count_nonerror + count_milderror + count_severeerror
prob_nonerror_mobinc = mobinc_nonerror / countall
prob_nonerror_tiemobred = tiemobred_nonerror / countall
prob_nonerror_mobred = mobred_nonerror / countall
prob[0,0]=prob_nonerror_mobinc[1]
prob[0,1]=prob_nonerror_tiemobred[1]
prob[0,2]=prob_nonerror_mobred[1]

prob_milderror_mobinc = mobinc_milderror / countall
prob_milderror_tiemobred = tiemobred_milderror / countall
prob_milderror_mobred = mobred_milderror / countall
prob[1,0]=prob_milderror_mobinc[1]
prob[1,1]=prob_milderror_tiemobred[1]
prob[1,2]=prob_milderror_mobred[1]

prob_severeerror_mobinc = mobinc_severeerror / countall
prob_severeerror_tiemobred = tiemobred_severeerror / countall
prob_severeerror_mobred = mobred_severeerror / countall
prob[2,0]=prob_severeerror_mobinc[1]
prob[2,1]=prob_severeerror_tiemobred[1]
prob[2,2]=prob_severeerror_mobred[1]

print(prob)
print(np.sum(prob, axis=0))
print(np.sum(prob, axis=1))

print('\nChi squared contingency')
print(stats.chi2_contingency(prob * countall[1]))
print('\n observed freq')
print(prob * countall[1])

f, ax = plt.subplots(figsize=(6,5))
sns.heatmap(prob, annot=True, fmt="f", linewidths=.5, ax=ax, cmap=sns.color_palette("Blues"))
plt.xticks([0.5, 1.5, 2.5], ['Mob+', 'Mob0', 'Mob-'])
plt.yticks([0.5, 1.5, 2.5], ['NonError', 'MildError', 'SevereError'])
plt.suptitle('Probability of Different Errors')
plt.title('Normalized Overall')
plt.savefig(out_file3)
plt.close()



sys.exit()

