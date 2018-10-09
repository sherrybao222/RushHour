import json, operator

json_file = '/Users/chloe/Documents/RushHour/data/paths.json'
data = []
subject_list = []
subject_dict = {}
# load raw file
with open(json_file) as f:
	for line in f:
		data.append(json.loads(line))
# filter file
for i in range(0, len(data)):
	line = data[i]
	subject = line['instance']
	if subject in subject_dict:
		continue	
	else:
		subject_dict[subject] = True
		subject_list.append(subject)
# sort entries
len_dict = {}
sorted_dict = {}
for i in range(0, len(data)):
	line = data[i]
	subject = line['instance']
	opt_len = line['optimal_length']
	len_dict[subject] = int(opt_len)
sorted_dict = sorted(len_dict, key=len_dict.get)
#sorted_dict = sorted(len_dict.items(), key=lambda x:x[1])
print(subject_list)
print(len(subject_list))
print(sorted_dict)
print(len(sorted_dict))

