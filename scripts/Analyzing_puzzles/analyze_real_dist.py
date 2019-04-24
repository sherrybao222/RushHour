from sys import argv
from analyze import *

"""
need to run with py27
calculate true distance to goal and all optimal paths
command:
python analyze_real_dist.py /Users/chloe/Documents/RushHour/exp_data/trialdata_valid.csv -1
"""
try:
    filename = argv[1]
    index = int(argv[2])
except:
    print 'usage: analyize_real_dist.py <filename> <index>'
calc_real_dist(filename,index)
