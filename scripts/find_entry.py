import json

json_file = '/Users/chloe/Documents/all_stages/paths.json'
data = []
subject_list = []
subject_dict = {}
with open(json_file) as f:
	for line in f:
		data.append(json.loads(line))

for i in range(0, len(data)):
	line = data[i]
	subject = line['instance']
	if subject in subject_dict:
		continue	
	else:
		subject_dict[subject] = True
		subject_list.append(subject)

print(subject_list)
print(len(subject_list))

