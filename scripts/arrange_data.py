# rearrange name of data in solution data folder
import os
from shutil import copyfile
# change to main directory
main_directory = '/Users/chloe/Documents/RushHour/exp_data/'
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
#main_directory = '/Users/chloe/Documents/test10/'
# os.chdir(main_directory)
# get all subject folders in main directory
all_file_names = os.listdir(main_directory + 'json/')
# iterate through all subjects to arrange files
for sub in all_file_names:
	for i in range(0, len(all_instances)):
		ins = all_instances[i]
		insname = '_' + ins + '_'
		# print(insname)
		# print(sub)
		if (insname in sub):
			# print('true')
			copyfile(main_directory + 'json/' + sub, main_directory + 'data_adopted/' + ins + '.json')
			print('rename succe: ' + sub)