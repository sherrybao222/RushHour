# combining all visualization and MAG info
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import MaxNLocator
import numpy as np

# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
datafile = '/Users/chloe/Documents/RushHour/figures/'
out_file = '/Users/chloe/Documents/RushHour/figures/'

# iterate through each puzzle
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	sub_dir = datafile + instance + '/'
	txt_file = sub_dir + instance + '.png'
	# MAG_info_file = sub_dir + instance + '_MAG_info.png'
	rss_file = sub_dir + instance + '_rss_count.png'
	time_file = sub_dir + instance + '_hum_time_distr.png'
	length_file = sub_dir + instance + '_hum_len_distr.png'
	MAG_file = sub_dir + instance + '_MAG.png'
	sub_out = sub_dir + instance + '_all_fig.png'

	fig = plt.figure(figsize=(24, 12))#,frameon=False)
	grid = plt.GridSpec(24, 48, wspace=0, hspace=0)
	
	txt = fig.add_subplot(grid[0:10, 0:10],xticklabels=[],yticklabels=[])
	MAG_info = fig.add_subplot(grid[0:10, 10:40],xticklabels=[],yticklabels=[])
	rss = plt.subplot(grid[10:, 0:12],xticklabels=[],yticklabels=[])
	time = plt.subplot(grid[10:, 12:26],xticklabels=[],yticklabels=[])
	length = plt.subplot(grid[10:, 26:40],xticklabels=[],yticklabels=[])
	MAG = plt.subplot(grid[:, 40:],xticklabels=[],yticklabels=[])
	
	txt.imshow(mpimg.imread(txt_file))
	# MAG_info.imshow(mpimg.imread(MAG_info_file))
	rss.imshow(mpimg.imread(rss_file))
	time.imshow(mpimg.imread(time_file))
	length.imshow(mpimg.imread(length_file))
	MAG.imshow(mpimg.imread(MAG_file))
	
	# txt.tick_params(bottom=False,top=False,left=False,right=False)
	# MAG_info.tick_params(bottom=False,top=False,left=False,right=False)
	# rss.tick_params(bottom=False,top=False,left=False,right=False)
	# time.tick_params(bottom=False,top=False,left=False,right=False)
	# length.tick_params(bottom=False,top=False,left=False,right=False)
	# MAG.tick_params(bottom=False,top=False,left=False,right=False)

	txt.axis('off')
	MAG_info.axis('off')
	rss.axis('off')
	time.axis('off')
	length.axis('off')
	MAG.axis('off')

	plt.tight_layout(pad=4, h_pad=0, w_pad=0, rect=None) # white space to figure edges
	# plt.show()
	plt.savefig(sub_out)
	plt.close()
