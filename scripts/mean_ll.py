''' 
BFS with restart and surrender model self-defined ll function,
speeded version, prepared for BADS in MATLAB,
python3 or py27
'''
import random, copy, pickle, os, sys, time
from operator import attrgetter
import multiprocessing as mp
import numpy as np
from numpy import recfromcsv
from json import dump, load
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
import scipy.stats as stats
from sklearn.model_selection import KFold
from Car import *
from Board import *
from Node import *



from BFS import *
def mean_ll(positions, decisions, params):
	pool = mp.Pool(processes=mp.cpu_count())
	print(ibs_early_stopping(positions, decisions, params, pool)/len(positions))
	pool.join()
	pool.close()

if __name__ == "__main__":
	positions = pickle.load(open('/Users/chloe/Desktop/carlists/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9_positions.pickle', 'rb'))[:300]
	decisions = pickle.load(open('/Users/chloe/Desktop/carlists/A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9_decisions.pickle', 'rb'))[:300]
	params = Params() # TODO
	mean_ll(positions, decisions, params)