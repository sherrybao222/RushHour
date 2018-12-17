# save new move json data by filtering out invalid subjects
# create new move file for each puzzle
# invalid subject: 1. surrender within 7 moves for every trial; 2. incomplete at the end
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import MAG
from scipy import stats

movefile = '/Users/chloe/Documents/RushHour/exp_data/moves.json'
outmove = '/Users/chloe/Documents/RushHour/exp_data/moves_filtered.json'
outfile = '/Users/chloe/Documents/RushHour/exp_data/'
# sorted according to optimal length
all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
out_data = []
out_file = []
# valid includes bonus and postquestionare
valid_subjects = ['ARWF605I7RWM7:3AZHRG4CU5SYU12YRVPCSZALW5003R', \
					'A289D98Z4GAZ28:3ZV9H2YQQEFR2R3JK2IXZUJPXAF3WW', \
					'A191V7PT3DQKDP:3PMBY0YE28B43VMUKKJ6EDF85RV9C8', \
					'A2MYB6MLQW0IGN:3TYCR1GOTDRCCQYD1V64UK7OE48LZS', \
					'A214HWAW1PYWO8:34BBWHLWHBJ6SUL255PK30LEGY5IWG', \
					'A18QU0YQB6Q8DF:3VE8AYVF8N5BS2NU6U3TMN50JMNF8V', \
					'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB', \
					'A21S7MA9DCW95E:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A2RTJ6BDY0DZVZ:3IJXV6UZ1YR1KY4G6BFEG1DXP7SRI1', \
					'A3NMQ3019X6YE0:3I2PTA7R3U2SESF4TZBQORI5LSJQKK', \
					'A18QU0YQB6Q8DF:3VE8AYVF8N5BS2NU6U3TMN50JMNF8V', \
					'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB', \
					'A21S7MA9DCW95E:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A2RTJ6BDY0DZVZ:3IJXV6UZ1YR1KY4G6BFEG1DXP7SRI1', \
					'A3NMQ3019X6YE0:3I2PTA7R3U2SESF4TZBQORI5LSJQKK', \
					'A15781PHGW377Y:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I', \
					'A2IBQA01NR5N76:3X4MXAO0BHWJLTOLVSJTHSM54OHWRG', \
					'A23437BMZ5T1FH:3IHR8NYAM89M0EPM8U9LH53ZJDTP4U', \
					'A18QU0YQB6Q8DF:3VE8AYVF8N5BS2NU6U3TMN50JMNF8V', \
					'A2TQNX64349OZ9:3WSELTNVR4AZUVYAYCSWZIQW0J4ATB', \
					'A21S7MA9DCW95E:3X87C8JFV7JQ2BSCY8KSFD9F2ILSQK', \
					'A2RTJ6BDY0DZVZ:3IJXV6UZ1YR1KY4G6BFEG1DXP7SRI1', \
					'A3NMQ3019X6YE0:3I2PTA7R3U2SESF4TZBQORI5LSJQKK', \
					'A15781PHGW377Y:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I', \
					'A2IBQA01NR5N76:3X4MXAO0BHWJLTOLVSJTHSM54OHWRG', \
					'A23437BMZ5T1FH:3IHR8NYAM89M0EPM8U9LH53ZJDTP4U', \
					'A1LR0VQIHQUJAM:3YW4XOSQKRTI0K0Z2YPDTDJVFABU1I', \
					'A1XDMS0KFSF5JW:3M0BCWMB8W4W5M7WZVX3HDH1M3FBWL', \
					'A15FXHC1CVNW31:3TS1AR6UQRM7SOIBWPBN8N9589Y7FV', \
					'A30KYQGABO7JER:37TD41K0AIHM8AITTQJXV8KY05QCSB', \
					'A3NMQ3019X6YE0:3V5Q80FXIYZ5QB5C6ITQBN30WY823U', \
					'A13BZCNJ0WR1T7:3TYCR1GOTDRCCQYD1V64UK7OHTBZLQ', \
					'A1EY7WONSYGBVY:3O6CYIULEE9B1LG2ZMEYM39PDJKUWB', \
					'A1AKX1C8GCVCTP:3H0W84IWBLAP4T2UASPNVMF5ZH7ER9', \
					'A3GXC3VG37CQ3G:3TPWUS5F8A9FFRZ2DVTYSXNJ6BWWC6', \
					'A3CTXNQ2GXIQSP:34HJIJKLP64Z5YMIU6IKNXSH7PDV4I', \
					'A3BPRPN10HJD4B:3TXD01ZLD5PZSJXIPG8FRBQYTI9U4X', \
					'A23XJ8I86R0O3B:30IQTZXKALEAAZ9CBKW0ZFZP6O5X07', \
					'A3OB6REG5CGHK1:3STRJBFXOXZ5687WA35LTWTS7QZKTT', \
					'A3Q0XAGQ7TD7MP:3GLB5JMZFY3TNXFGYMKRQ0JDXNTDGZ', \
					'AXKM02NVXNGOM:33PPO7FECWN7JOLBOAKUBCWTC0AIDL', \
					'A1USR9JCAMDGM3:3PB5A5BD0WED6OE679H5Q89HBDGG71', \
					'A1F4N58CAX8IMK:35DR22AR5ES6RR89U7EJ1DXW91AX3P', \
					'A1N1EF0MIRSEZZ:3R5F3LQFV3SKIB1AENMWM1BICT5OZB', \
					'A1GQS6USF2JEYG:33F859I567LE8WC74WB3GA7E94XBHW', \
					'A18T3WK7J16C1B:3IJXV6UZ1YR1KY4G6BFEG1DXR47RIC', \
					'A1ZTSCPETU3UJW:3XC1O3LBOTUGQEPEV3HM8W67X4RTLL', \
					'A2RCYLKY072XXO:3TAYZSBPLMG9ASQRWXURJVBCPLN2SH', \
					'AO1QGAUZ85T6L:3HWRJOOET6A15827PHPSLWK1MOYSEU', \
					'A3GOOI75XOF24V:3TMFV4NEP9MD3O9PWJDTQBR0HZYW8I', \
					'A30AGR5KF8IEL:3EWIJTFFVPF14ZIVGF68BQEIRXP0ER', \
					'A2QIZ31TMHU0GD:3FUI0JHJPY6UBT1VAI7VUX8S3KH33W', \
					'AMW2XLD9443OH:37QW5D2ZRHUKW7SGCE3STMOFBMK8SX', \
					'A53S7J4JGWG38:3M0BCWMB8W4W5M7WZVX3HDH1P53WB1', \
					'A28RX7L0QZ993M:3SITXWYCNWHBUMCM90TPJWV8Y7AXBW', \
					'ALH1K6ZAQQMN7:3IXEICO793RY7TM78ZBKJDOA7ADT6Q']
print('valid subjects len', len(valid_subjects))


def sort_by_sub(d):
    '''a helper function for sorting'''
    return d['subject']
def sort_by_ins(d):
    '''a helper function for sorting'''
    return d['instance']

# process move json file
old_move_data = []
# read in data
with open(movefile) as f: # read path data
	for line in f:
		old_move_data.append(json.loads(line))
# sort by subject
old_move_data = sorted(old_move_data, key=sort_by_sub)
new_move_data_tmp = []
new_move_data = [] # outmove
# iterate through line
cur_sub = ''
prev_sub = ''
surrender_error = True
incomplete_error = True
sub_data = []
for i in range(0, len(old_move_data)): # line
	line = old_move_data[i]
	instance = line['instance']
	subject = line['subject']
	move_number = int(line['move_number'])
	cur_sub = subject
	if move_number >= 7:
		surrender_error = False
	if cur_sub != prev_sub and (prev_sub != ''): 
	# begin a new subject, except for the first one
		if (not surrender_error) and (not incomplete_error): # save or not
			new_move_data_tmp.append(sub_data)
		# clean data field for the next subject
		sub_data = []
		surrender_error = True
		incomplete_error = True
	sub_data.append(line)
	if i == len(old_move_data) - 1: # at the end of file
		if (not surrender_error) and (not incomplete_error):
			new_move_data_tmp.append(sub_data)
	prev_sub = subject
# flatten the move file
for sub in new_move_data_tmp:
	for move in new_move_data_tmp:
		new_move_data.append(move)
# sort the new move file by instance 
new_move_data = sorted(new_move_data, key=sort_by_ins)
# save the new moves file
with open(outmove, 'w+') as f:
	for i in range(0, len(new_move_data)): # each line
		json.dump(new_move_data[i], f)
		f.write('\n')


# process new move files for each puzzle
# initialize lists
for i in range(0, len(all_instances)):
	instance = all_instances[i]
	out_file.append(outfile + instance + '_moves_filtered.json')
	out_data.append([])
# read every line of new move data
for line in new_move_data:
	cur_line = json.loads(line)
	cur_ins = cur_line['instance']
	ins_index = all_instances.index(cur_ins)
	# print(ins_index)
	# append current line to corresponding list index
	cur_data = out_data[ins_index]
	cur_data.append(cur_line)
	out_data[ins_index] =  cur_data
# write to file
for i in range(0, len(all_instances)):
    cur_data = out_data[i]
    cur_data = sorted(cur_data, key=sort_by_sub) # sort data by subject
    cur_file = out_file[i]
    cur_file = open(cur_file, 'w+')
    for j in range(0, len(cur_data)):
	    json.dump(cur_data[j], cur_file)
	    cur_file.write('\n')
	    