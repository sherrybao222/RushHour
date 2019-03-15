# all functions for preprocessing moves data for each subject trial (valid)
import json
import numpy as np
import csv
import sys
from collections import namedtuple, defaultdict
import re # regular expression
import os

PsiturkRec = namedtuple('PsiturkRec', 'worker assignment ord t event piece move_nu move instance')
PsiturkRec.__new__.__defaults__ = (None,) * len(PsiturkRec._fields)

# read raw data csv moves file trialdata.csv
# need to run with python2.7
def read_psiturk_data(filename):
    recs = [] # out data
    # count the number of valid subjects found
    sub_flag = [False] * len(valid_sub)
    with open(filename,'r') as log:
        for l in log:
            if 'event' not in l: # instructions phase
                continue
            sub_name = l.split(',')[0] # subject name
            if sub_name not in valid_sub:
                continue
            else:
                sub_flag[list(valid_sub).index(sub_name)] = True
            v = l.split(',')[0].split(':') # append sub name
            v.append(l.split(',')[1]) # subject data number
            # read move data fields
            v += [s.replace('[','').replace(']','') for s in re.findall('\[.*?\]',l)]
            recs.append(PsiturkRec(*v))
        print 'valid subject found'
        print sum(sub_flag)
    return recs


# preprocess raw data csv moves file to formatted csv moves file
# need to run with python2.7
def prep_trialdata(orig_filename, dest_filename):
    
    print 'subject, assignment, instance, optlen, move_num, move, meta_move, rt, trial_number\n'
    
    # read trialdata file
    recs = read_psiturk_data(orig_filename)
    
    # load optlen, each puzzle's optlen is part of its config filename
    json_dir = '/Users/chloe/Documents/RushHour/psiturk-rushhour/static/json'
    jsons = os.listdir(json_dir)
    jsons = [j for j in jsons if j.endswith('.json')]
    jsons = dict([(j.split('_')[2],j.split('_')[3]) for j in jsons]) 


    # start iterating data
    with open(dest_filename, mode = 'w') as outfile:
        filewriter = csv.writer(outfile, delimiter=',')
        filewriter.writerow(['worker', 'assignment', 'instance', 'optlen', 'move_number', 'move (car@pos)', 'meta_move', 'rt (ms)', 'trial_number'])
        trial_number = 0
        for r in recs:
            ins = r.instance
            # optlen
            try:
                opt_solution = jsons[r.instance]
            except:
                opt_solution = 999
            # move number in this trial
            move_nu = r.move_nu
            # current move
            if r.event == 'start': # if the first move in this trial
                trial_number += 1
                is_last_win = False
                initial_time = r.t
            if r.event == 'drag_end' or r.event == 'win' : # after each move
                move = '{0}@{1}'.format(r.piece,r.move).replace(',','.').replace(' ','')
                meta_move = r.event
                if r.event == 'win':
                    if is_last_win:
                        continue
                    is_last_win = True
                # response time for current move
                rt = float(r.t) - float(initial_time)
                initial_time = r.t
                # all fields in this move
                fields = [r.worker, r.assignment, ins, opt_solution, move_nu, move, meta_move, rt, trial_number]
                filewriter.writerow(','.join(['{}']*len(fields)).format(*fields).split(','))
                # reset trial number if win
                if r.event == 'win':
                    trial_number = 0
                # sys.exit()
    


# main execution calls
original_file = '/Users/chloe/Documents/RushHour/exp_data/trialdata.csv'
dest_file = '/Users/chloe/Documents/RushHour/exp_data/moves_valid.csv'
valid_sub = np.load('/Users/chloe/Documents/RushHour/exp_data/valid_sub.npy')
prep_trialdata(original_file, dest_file)
