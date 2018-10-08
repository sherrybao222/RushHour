import csv, json

csv_file = '/Users/chloe/Documents/all_stages/moves.csv'
json_file = '/Users/chloe/Documents/all_stages/moves.json'

'''
with open(data_dir) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'subject \t{row[0]} instance {row[1]} opt_len {row[2]} move_num {row[3]} move {row[4]} distance to goal {row[10]}.')
            line_count += 1
    print(f'Processed {line_count} lines.')
'''
csvfile = open(csv_file, 'r')
jsonfile = open(json_file, 'w+')
fieldnames = ("subject","instance","optimal_length","move_number",  "move", "pre_actions", "meta_move", "rt", "trial_number", "progress", "distance_to_goal")
reader = csv.DictReader(csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')


csv_file = '/Users/chloe/Documents/all_stages/paths.csv'
json_file = '/Users/chloe/Documents/all_stages/paths.json'
csvfile = open(csv_file, 'r')
jsonfile = open(json_file, 'w+')
fieldnames = ("subject","assignment","instance","optimal_length", "human_length", "complete", "start_time", "end_time", "rt", "nodes_expanded", "skipped", "trial_number")
reader = csv.DictReader(csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
