'''
plot summary statistics of fitting results
distance to goal as a function of move number
divided by level
compare data, model, and random
one subject only (need to generalize to all subjetcs)
python 3
'''
import pandas as pd
from BFS import *

sub_path = '/Users/chloe/Desktop/subjects/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9.csv'
fit_params = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 0.01]
random_params = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
				0.01, 10, 1.0] # lapse rate = 1.0

# load subject data
subdata = pd.read_csv(sub_path)
level7 = df[(df.initial==7)]
