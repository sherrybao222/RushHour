# linear regression on MAG info to human_len
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
features = ['y_opt',\
        'y_unsafesol', 'y_backmovesol', \
        'y_avgnodesol', 'y_avgedgesol', \
        'y_avgncycle', 'y_avgmaxcycle', \
        'y_avgcnode', 'y_avgdepth',\
        'y_noderate', 'y_edgerate',
        'y_unsafe_human', 'y_backmove_human', \
        'y_avgnode_human', 'y_avgedge_human', \
        'y_avgncycle_human', 'y_avgmaxcycle_human', \
        'y_avgcnode_human', 'y_avgdepth_human']
feature_labels = ['opt_len',\
      'p_unsafe_sol', 'p_backmove_sol', \
      'avg_node_sol', 'avg_edge_sol', \
      'avg_ncycle_sol', 'avg_maxcycle',\
      'avg_node_incycle', 'avg_depth',\
      'node_rate', 'edge_rate', \
      'p_unsafe_human', 'p_backmove_human', \
      'avg_node_human', 'avg_edge_human', \
      'avg_ncycle_human', 'avg_maxcycle_human',\
      'avg_node_incycle_human', 'avg_depth_human']
targets = ['y_human']
data_dir = '/Users/chloe/Documents/RushHour/state_model/in_data2/'
out_dir = '/Users/chloe/Documents/RushHour/state_model/out_model/'
fig_out = '/Users/chloe/Documents/RushHour/state_model/out_model/linear_regr6.png'
coef_out = '/Users/chloe/Documents/RushHour/state_model/out_model/linear_regr_coef6.png'

# initialize feature data, [70,1]
feature_data = np.expand_dims(np.load(data_dir + features[0] + '.npy'), axis=1) 
for i in range(1, len(features)):
	d = features[i]
	cur_data = np.expand_dims(np.load(data_dir + d + '.npy'), axis=1) # [70,1]
	feature_data = np.concatenate((feature_data, cur_data), axis=1) # [70, n_features]
print('feature shape: ', feature_data.shape)
feature_data = feature_data.astype(np.float64)
# feature_data: [n_samples 70, n_features]

# initialize target data #[70,1]
target_data = np.expand_dims(np.load(data_dir + targets[0] + '.npy'), axis=1)
# print('target shape: ', target_data.shape)
for i in range(1, len(targets)):
	t = targets[i]
	cur_data = np.expand_dims(np.load(data_dir + t + '.npy'), axis=1) #[70,1]
	target_data = np.concatenate((target_data, cur_data), axis=1) #[70,n_targets]
print('target shape: ', target_data.shape)
target_data = target_data.astype(np.float64)

# linear regression
regr = linear_model.LinearRegression(normalize=True)
regr.fit(feature_data, target_data)
predict = regr.predict(feature_data) # [70, 1]
R2 = r2_score(target_data, predict)
AdjR2 = 1 - (1 - R2) * (predict.shape[0] - 1) / (predict.shape[0] - len(features) - 1)
print('Coefficients: \n' + str(regr.coef_)) # coefficients
print("Mean squared error: %.2f" \
      % mean_squared_error(target_data, predict)) # mean squared error
print('Variance score R2: %.2f' \
		% R2)# Explained variance perfect prediction
print('Adjusted R2: %.2f' \
		% AdjR2)# Explained variance perfect prediction

# plot outputs
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111)
ax.scatter(np.arange(len(all_instances)), target_data.T, s=12, alpha=0.8, color='orangered')
ax.scatter(np.arange(len(all_instances)), predict.T, s=12, alpha=0.8, color='green')
ax.plot(np.arange(len(all_instances)), np.squeeze(target_data.T), alpha=0.8, color='orangered', label='target')
ax.plot(np.arange(len(all_instances)), np.squeeze(predict.T), alpha=0.8, color='green', label='predict')
ax.set_xticks([18,36,53,70])
ax.set_xlabel('all puzzles')
ax.set_ylabel('len')
ax.legend(loc='upper left')
ax.grid(axis = 'x', alpha = 0.3)
plt.suptitle('Linear regression of dynamic MAG to human_len', fontweight='bold')
plt.title("Mean squared error: %.2f" % mean_squared_error(target_data, predict)\
			+ ', R2: %.2f' % R2 + ', AdjR2: %.2f' % AdjR2, fontsize=10)
# plt.show()
plt.savefig(fig_out)
plt.close()

# fig, ax = plt.subplots()
plt.figure(figsize=(9,10))
plt.scatter(np.arange(len(features)), regr.coef_)
plt.xticks(np.arange(len(features)),feature_labels, rotation=45)
plt.grid(axis = 'y', alpha = 0.3)
plt.grid(axis = 'x', alpha = 0.3)
plt.suptitle('Linear regression coefficients', fontweight='bold')
plt.title("Mean squared error: %.2f" % mean_squared_error(target_data, predict)\
			+ ', R2: %.2f' % R2 + ', AdjR2: %.2f' % AdjR2, fontsize=10)
plt.savefig(coef_out)
plt.close()

'''
1
['y_nodes','y_edges', 'y_en', 'y_enp', 'y_e2n', \
'y_countscc', 'y_maxscc', \
'y_countcycle', 'y_maxcycle', 'y_c_incycle', 'y_nnc', \
'y_pnc', 'y_depth', 'y_ndepth']
Coefficients:
[[ 1.14884410e+00  7.57834721e+00 -7.42818550e+01  4.50304404e+00
  -3.55845950e+00  1.51312688e-01 -7.84489903e-01 -5.08583139e-02
   1.41263361e-01 -6.96560632e+00  6.51317078e+01  2.53844950e+00
   4.73463738e-01]]
Mean squared error: 146.04
Variance score R2: 0.23
'''
'''
2
['y_opt', 'y_nodes','y_edges', 'y_en', 'y_enp', 'y_e2n', \
'y_countscc', 'y_maxscc', \
'y_countcycle', 'y_maxcycle', 'y_c_incycle', 'y_nnc', \
'y_pnc', 'y_depth', 'y_ndepth']
Coefficients:
[[ 3.16397184e+00 -7.79410147e+00  7.59809802e+00 -7.56373409e+01
   1.73060553e+00 -1.30510233e+00  1.38180539e+00 -4.65386503e-01
   2.39784345e-01  3.41350016e-02 -1.22839927e+00 -2.50966475e-01
   1.79294140e+00  3.28737434e-01]]
Mean squared error: 48.83
Variance score R2: 0.74
'''


'''
3
['y_nodes','y_edges', 'y_en', 'y_enp', 'y_e2n', \
'y_countscc', 'y_maxscc', \
'y_countcycle', 'y_maxcycle', 'y_c_incycle', 'y_nnc', \
'y_pnc', 'y_depth', 'y_ndepth', \
'y_gcluster', 'y_lcluster']
Coefficients:
[[ 4.99394617e+01 -3.95365115e+01  1.10977722e+02  3.45987678e+00
   8.11810655e+00 -2.72684304e+00  9.93320747e-01 -8.52723033e-01
  -1.63323004e-01  9.73636880e-02 -4.46036534e+00  3.83785952e+01
   3.59780672e+00  6.70326673e-01 -2.88006408e+01  2.27430556e+01]]
Mean squared error: 142.34
Variance score R2: 0.25
Adjusted R2: 0.02
'''
'''
4
['y_opt', \
'y_nodes','y_edges', 'y_en', 'y_enp', 'y_e2n', \
'y_countscc', 'y_maxscc', \
'y_countcycle', 'y_maxcycle', 'y_c_incycle', 'y_nnc', \
'y_pnc', 'y_depth', 'y_ndepth', \
'y_gcluster', 'y_lcluster']
Coefficients:
[[ 3.19624737e+00 -6.24259083e+00  7.80850586e+00 -3.61127131e+01
   3.31159015e+00 -1.50843235e+00 -2.74246965e-01  1.59560937e+00
  -2.34394025e-01 -3.31802352e-02  5.13274371e-02  1.04711544e+00
  -2.26382461e+01  1.98615074e+00  1.22659861e-01 -3.57763674e+01
   3.26432365e+01]]
Mean squared error: 46.92
Variance score R2: 0.75
Adjusted R2: 0.67
'''
'''
5
MAGs -> opt_len
Coefficients:
[[ 1.75775045e+01 -1.48126887e+01  4.60197283e+01  4.63939766e-02
   3.01182536e+00 -7.67336127e-01 -1.88436173e-01 -1.93454679e-01
  -4.07173644e-02  1.44032190e-02 -1.72310843e+00  1.90901499e+01
   5.04233806e-01  1.71346817e-01  2.18247394e+00 -3.09743888e+00]]
Mean squared error: 9.34
Variance score R2: 0.19
Adjusted R2: -0.05
'''


'''
6
updated global cc
['y_nodes','y_edges', 'y_en', 'y_enp', 'y_e2n', \
'y_countscc', 'y_maxscc', \
'y_countcycle', 'y_maxcycle', 'y_c_incycle', 'y_nnc', \
'y_pnc', 'y_depth', 'y_ndepth', \
'y_gcluster', 'y_lcluster']
Coefficients:
[[ 4.78715397e+01 -3.74607227e+01  8.07231923e+01  1.56382314e+00
   8.68262333e+00 -3.57821064e+00  8.08618388e-01 -9.93238525e-01
   1.17105568e-02  8.97457541e-02 -6.34780289e+00  5.58889357e+01
   3.03842911e+00  6.58906777e-01  3.04094912e+01 -5.74824071e+01]]
Mean squared error: 142.99
Variance score R2: 0.24
Adjusted R2: 0.02
'''
'''
7
updated global cc
['y_opt','y_nodes','y_edges', 'y_en', 'y_enp', 'y_e2n', \
'y_countscc', 'y_maxscc', \
'y_countcycle', 'y_maxcycle', 'y_c_incycle', 'y_nnc', \
'y_pnc', 'y_depth', 'y_ndepth', \
'y_gcluster', 'y_lcluster']
[[ 3.18209538e+00 -1.08492522e+01  1.18338764e+01 -8.23684900e+01
   1.37590254e+00 -1.00627245e+00 -1.29711118e+00  1.35058112e+00
  -4.24304050e-01  2.12336240e-01  4.23345858e-02 -1.24364124e+00
  -8.85047108e-01  1.43999372e+00  1.93118065e-01  2.43251973e+01
  -4.15518315e+01]]
Mean squared error: 48.37
Variance score R2: 0.74
Adjusted R2: 0.66
'''


####################### DYNAMIC MAG #####################
'''
1
features = [\
        'y_unsafesol', 'y_backmovesol', \
        'y_avgnodesol', 'y_avgedgesol', \
        'y_avgncycle', 'y_avgmaxcycle', \
        'y_avgcnode', 'y_avgdepth',\
        'y_noderate', 'y_edgerate']
feature_labels = [\
      'p_unsafe_sol', 'p_backmove_sol', \
      'avg_node_sol', 'avg_edge_sol', \
      'avg_ncycle_sol', 'avg_maxcycle',\
      'avg_node_incycle', 'avg_depth',\
      'node_rate', 'edge_rate']
Coefficients:
[[ -8.35750211 -12.6311428   -0.24451505   2.84487233  -3.95906924
   -4.56138819   3.96136665  -4.39512972 -22.73922662  -2.60151809]]
Mean squared error: 45.08
Variance score R2: 0.76
Adjusted R2: 0.72
'''
'''
2
features = ['y_opt',\
        'y_unsafesol', 'y_backmovesol', \
        'y_avgnodesol', 'y_avgedgesol', \
        'y_avgncycle', 'y_avgmaxcycle', \
        'y_avgcnode', 'y_avgdepth',\
        'y_noderate', 'y_edgerate']
feature_labels = ['opt_len',\
      'p_unsafe_sol', 'p_backmove_sol', \
      'avg_node_sol', 'avg_edge_sol', \
      'avg_ncycle_sol', 'avg_maxcycle',\
      'avg_node_incycle', 'avg_depth',\
      'node_rate', 'edge_rate']
Coefficients:
[[  2.0848726   -4.73947115  -3.64014921  -0.87900273   2.48055209
   -1.67874709  -2.74019751   1.42321392  -2.34325258  11.93450136
  -10.24967195]]
Mean squared error: 42.04
Variance score R2: 0.78
Adjusted R2: 0.74
'''
'''
3
features = [\
        'y_unsafesol', 'y_backmovesol', \
        'y_avgnodesol', 'y_avgedgesol', \
        'y_avgncycle', 'y_avgmaxcycle', \
        'y_avgcnode', 'y_avgdepth',\
        'y_noderate', 'y_edgerate',
        'y_unsafe_human', 'y_backmove_human', \
        'y_avgnode_human', 'y_avgedge_human', \
        'y_avgncycle_human', 'y_avgmaxcycle_human', \
        'y_avgcnode_human', 'y_avgdepth_human']
feature_labels = [\
      'p_unsafe_sol', 'p_backmove_sol', \
      'avg_node_sol', 'avg_edge_sol', \
      'avg_ncycle_sol', 'avg_maxcycle',\
      'avg_node_incycle', 'avg_depth',\
      'node_rate', 'edge_rate', \
      'p_unsafe_human', 'p_backmove_human', \
      'avg_node_human', 'avg_edge_human', \
      'avg_ncycle_human', 'avg_maxcycle_human',\
      'avg_node_incycle_human', 'avg_depth_human']
Coefficients:
[[-15.99182245 -52.62089827  -4.32876618   3.39714672  -5.91215302
   -6.06777242   6.43303974  -7.66789061 -21.15206775  -5.60754913
   -0.12100117  55.92024714   7.46767513  -0.463913    -0.89142561
    4.3712246   -5.93363869   3.4448908 ]]
Mean squared error: 34.47
Variance score R2: 0.82
Adjusted R2: 0.75
'''
'''
4
features = ['y_opt',\
        'y_unsafesol', 'y_backmovesol', \
        'y_avgnodesol', 'y_avgedgesol', \
        'y_avgncycle', 'y_avgmaxcycle', \
        'y_avgcnode', 'y_avgdepth',\
        'y_noderate', 'y_edgerate',
        'y_unsafe_human', 'y_backmove_human', \
        'y_avgnode_human', 'y_avgedge_human', \
        'y_avgncycle_human', 'y_avgmaxcycle_human', \
        'y_avgcnode_human', 'y_avgdepth_human']
feature_labels = ['opt_len',\
      'p_unsafe_sol', 'p_backmove_sol', \
      'avg_node_sol', 'avg_edge_sol', \
      'avg_ncycle_sol', 'avg_maxcycle',\
      'avg_node_incycle', 'avg_depth',\
      'node_rate', 'edge_rate', \
      'p_unsafe_human', 'p_backmove_human', \
      'avg_node_human', 'avg_edge_human', \
      'avg_ncycle_human', 'avg_maxcycle_human',\
      'avg_node_incycle_human', 'avg_depth_human']
Coefficients:
[[  1.30563466 -11.41651824 -51.77173565  -3.79020072   3.02199734
   -3.8946467   -4.49035485   4.08749288  -6.52894489   1.57885103
  -10.5951934   -4.65255298  61.74038745   5.5078418    0.14448791
   -1.03570294   3.59884325  -5.36827396   3.56199139]]
Mean squared error: 33.55
Variance score R2: 0.82
Adjusted R2: 0.76
'''
'''
5
features = [\
        'y_noderate', 'y_edgerate',
        'y_unsafe_human', 'y_backmove_human', \
        'y_avgnode_human', 'y_avgedge_human', \
        'y_avgncycle_human', 'y_avgmaxcycle_human', \
        'y_avgcnode_human', 'y_avgdepth_human']
feature_labels = [\
      'node_rate', 'edge_rate', \
      'p_unsafe_human', 'p_backmove_human', \
      'avg_node_human', 'avg_edge_human', \
      'avg_ncycle_human', 'avg_maxcycle_human',\
      'avg_node_incycle_human', 'avg_depth_human']
Coefficients:
[[ -0.90445848 -19.65388311 -12.3122338   51.08436149   3.92259337
    0.7979805   -0.69945621   3.09220107  -2.6323386    2.56933987]]
Mean squared error: 43.43
Variance score R2: 0.77
Adjusted R2: 0.73
'''
'''
6
features = ['y_opt',\
        'y_noderate', 'y_edgerate',
        'y_unsafe_human', 'y_backmove_human', \
        'y_avgnode_human', 'y_avgedge_human', \
        'y_avgncycle_human', 'y_avgmaxcycle_human', \
        'y_avgcnode_human', 'y_avgdepth_human']
feature_labels = ['opt_len',\
      'node_rate', 'edge_rate', \
      'p_unsafe_human', 'p_backmove_human', \
      'avg_node_human', 'avg_edge_human', \
      'avg_ncycle_human', 'avg_maxcycle_human',\
      'avg_node_incycle_human', 'avg_depth_human']
Coefficients:
[[  2.43511737  33.25176718 -23.29261959 -14.41149184  54.20572374
    0.99720372   1.72148364  -0.80540924   2.77550151  -2.99016693
    2.79705801]]
Mean squared error: 39.13
Variance score R2: 0.79
Adjusted R2: 0.75
'''

