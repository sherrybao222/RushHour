# visualize puzzle from txt to image
import json, os, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
import text_to_image

datafile = '/Users/chloe/Documents/RushHour/figures/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_dir = '/Users/chloe/Documents/RushHour/figures/'

# iterate through each puzzle
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	ins_txt_dir = datafile + instance + '/' + instance + '.txt'
	ins_out_dir = out_dir + instance + '/' + instance + '.png'
	# encoded_image_path = text_to_image.encode_file(ins_txt_dir, ins_out_dir)
	txt = open(ins_txt_dir).read()
	txt = txt.replace(' ', '  ')
	txt = txt.replace('r', 'R')
	fig = plt.figure(figsize=(4,6))
	plt.text(0,0,txt,fontsize=38)#,fontweight='bold')
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')
	plt.savefig(ins_out_dir)
	plt.close()