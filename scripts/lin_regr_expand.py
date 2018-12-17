# linear regression on MAG info to human_len
# using expanded data: each trial from every subject instead of the subject average
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

all_instances = ['prb8786', 'prb11647', 'prb21272', 'prb13171', 'prb1707', 'prb23259', 'prb10206', 'prb2834', 'prb28111', 'prb32795', 'prb26567', 'prb14047', 'prb14651', 'prb32695', 'prb29232', 'prb15290', 'prb12604', 'prb20059', 'prb9718', 'prb29414', 'prb22436', 'prb62015', 'prb38526', 'prb3217', 'prb34092', 'prb12715', 'prb54081', 'prb717', 'prb31907', 'prb42959', 'prb79230', 'prb14898', 'prb62222', 'prb68910', 'prb33509', 'prb46224', 'prb47495', 'prb29585', 'prb38725', 'prb33117', 'prb20888', 'prb55384', 'prb6671', 'prb343', 'prb68514', 'prb29600', 'prb23404', 'prb19279', 'prb3203', 'prb65535', 'prb14485', 'prb34551', 'prb72800', 'prb44171', 'prb1267', 'prb29027', 'prb24406', 'prb58853', 'prb24227', 'prb45893', 'prb25861', 'prb15595', 'prb54506', 'prb48146', 'prb78361', 'prb25604', 'prb46639', 'prb46580', 'prb10166', 'prb57223']
features = [\
        'y_unsafeSol', 'y_backMoveSol', \
        'y_avgNodeSol', 'y_avgEdgeSol', \
        'y_avgnCycleSol', 'y_avgMaxCycleSol', \
        'y_avgcNodeSol', 'y_avgDepthSol',\
        'y_nodeRate', 'y_edgeRate']
feature_labels = [\
      'p_unsafe_sol', 'p_backmove_sol', \
      'avg_node_sol', 'avg_edge_sol', \
      'avg_ncycle_sol', 'avg_maxcycle',\
      'avg_node_incycle', 'avg_depth',\
      'node_rate', 'edge_rate']
targets = ['y_human']
data_dir = '/Users/chloe/Documents/RushHour/state_model/in_data2/'
out_dir = '/Users/chloe/Documents/RushHour/state_model/out_model/'
fig_out = '/Users/chloe/Documents/RushHour/state_model/out_model/linear_regr_exp1.png'
coef_out = '/Users/chloe/Documents/RushHour/state_model/out_model/linear_regr_exp_coef1.png'

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
fig = plt.figure(figsize=(19,7))
ax = fig.add_subplot(111)
ax.scatter(np.arange(target_data.shape[0]), target_data.T, s=12, alpha=0.8, color='orangered')
ax.scatter(np.arange(target_data.shape[0]), predict.T, s=12, alpha=0.8, color='green')
ax.plot(np.arange(target_data.shape[0]), np.squeeze(target_data.T), alpha=0.8, color='orangered', label='target')
ax.plot(np.arange(target_data.shape[0]), np.squeeze(predict.T), alpha=0.8, color='green', label='predict')
# ax.set_xticks([18,36,53,70])
ax.set_xlabel('all sub trials')
ax.set_ylabel('len')
ax.legend(loc='upper left')
ax.grid(axis = 'x', alpha = 0.3)
plt.suptitle('Linear regression of sub trial dMAG to human_len', fontweight='bold')
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