# convert csv file to json
import csv, json

csv_file = '/Users/chloe/Documents/all_stages/moves.csv'
json_file = '/Users/chloe/Documents/all_stages/moves.json'

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
