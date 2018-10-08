import json

json_file = '/Users/chloe/Documents/all_stages/paths.json'
data = []
instance_dict = {}
mean_dict = {}
all_instances = ['prb9718', 'prb47495', 'prb29585', 'prb38725', 'prb8786', 'prb11647', 'prb44171', 'prb1267', 'prb29414', 'prb29027', 'prb22436', 'prb21272', 'prb62015', 'prb24406', 'prb13171', 'prb58853', 'prb38526', 'prb24227', 'prb33117', 'prb3217', 'prb45893', 'prb1707', 'prb25861', 'prb34092', 'prb15595', 'prb54506', 'prb48146', 'prb12715', 'prb20888', 'prb78361', 'prb23259', 'prb55384', 'prb54081', 'prb6671', 'prb25604', 'prb717', 'prb343', 'prb31907', 'prb10206', 'prb2834', 'prb42959', 'prb79230', 'prb46639', 'prb68514', 'prb28111', 'prb32795', 'prb26567', 'prb14898', 'prb29600', 'prb23404', 'prb46580', 'prb14047', 'prb10166', 'prb62222', 'prb14651', 'prb32695', 'prb19279', 'prb3203', 'prb29232', 'prb68910', 'prb33509', 'prb15290', 'prb46224', 'prb12604', 'prb20059', 'prb65535', 'prb14485', 'prb57223', 'prb34551', 'prb72800']
# preprocess dict
for i in range(len(all_instances)):
	instance_dict[all_instances[i] + '_count'] = 0
	instance_dict[all_instances[i]+'_len'] = 0
	mean_dict[all_instances[i]] = 0
# load json data
with open(json_file) as f:
	for line in f:
		data.append(json.loads(line))
# iterate through line
for i in range(0, len(data)):
	line = data[i]
	instance = line['instance']
	complete = line['complete']
	human_len = int(line['human_length'])
	if not complete:
		continue
	instance_dict[instance + '_count'] = instance_dict[instance + '_count'] + 1
	instance_dict[instance + '_len'] = instance_dict[instance + '_len'] + human_len
# calculate mean
for i in range(len(all_instances)):
	if instance_dict[all_instances[i] + '_count'] == 0:
		mean_dict[all_instances[i]] = 0
		continue
	mean_dict[all_instances[i]] = instance_dict[all_instances[i] + '_len'] / instance_dict[all_instances[i] + '_count']

print(mean_dict)