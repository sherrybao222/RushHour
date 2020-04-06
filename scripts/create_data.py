import pandas as pd
import os
from BFS import *

def create_sub_csv(datapath='/Users/chloe/Desktop/trialdata_valid_true_dist7_processed.csv',
					outputpath='/Users/chloe/Desktop/subjects/',
					column_entries=['subject', 'ord', 'rt', 'event', 'piece', 'move_num', 'move', 'instance', 'optlen', 'initial', 'restart', 'surrender', 'mobility']):
	''' split all subjects data into individual subject data (csv) '''
	data = pd.read_csv(filepath_or_buffer=datapath)
	all_sub = data.subject.unique()
	data = data.filter(column_entries)
	for subject, df in data.groupby('subject'):
		print(subject)
		df.to_csv(outputpath+subject+'.csv', index = False)
	print(all_sub)


def create_sub_pickle(datapath='/Users/chloe/Desktop/subjects/',
					outputpath='/Users/chloe/Desktop/carlists/',
					instancepath='/Users/chloe/Documents/RushHour/exp_data/data_adopted/'):
	''' use indivisual subject data csv to create car lists '''
	for file in os.listdir(datapath):
		df = pd.read_csv(datapath+file)
		subject = df.iloc[0]['subject']
		positions = [] # list of carlist
		decisions = [] # list of carlist
		for idx, row in df.iterrows():
			if row['event'] == 'start':
				instance = row['instance']
				ins_file = instancepath+instance+'.json'
				cur_carlist = json_to_car_list(ins_file)
				continue
			if row['piece']=='r' and row['move']==16 and row['optlen']==1: # win
				continue
			piece = str(row['piece'])
			move_to = int(row['move'])
			positions.append(cur_carlist) # previous carlist
			cur_carlist, _ = move(cur_carlist, piece, move_to)
			decisions.append(cur_carlist)
		with open(outputpath+subject+'_positions.pickle', 'wb') as f:
			pickle.dump(positions, f)
		with open(outputpath+subject+'_decisions.pickle', 'wb') as f:
			pickle.dump(decisions, f)

# create_sub_pickle()
# pickle.load(open('/Users/chloe/Desktop/carlists/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9_decisions.pickle', 'rb'))






