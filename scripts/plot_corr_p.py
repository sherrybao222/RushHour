# spearman corr and p value for new MAG and old MAG with human/optimal len
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

new_corr = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_corr.npy')
new_p = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_p.npy')
old_corr = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_old_corr.npy')
old_p = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_old_p.npy')
out_dir = '/Users/chloe/Documents/RushHour/figures/corr_p.png'
# data: ['hum&#n+#e','opt&#n+#e','hum&#n','hum&#e','opt&#n','opt&#e']

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.scatter(np.arange(4), new_corr[2:], s=60, color='orange', label='new MAG corr')
new_p_label = []
old_p_label = []
for i in range(2,len(new_p)):
	p = new_p[i]
	if 0.01 < p <= 0.05:
		new_p_label.append('*')
	elif 0.001 < p <= 0.01:
		new_p_label.append('**')
	elif p <= 0.001:
		new_p_label.append('***')
for i in range(2,len(old_p)):
	p = old_p[i]
	if 0.01 < p <= 0.05:
		old_p_label.append('*')
	elif 0.001 < p <= 0.01:
		old_p_label.append('**')
	elif p <= 0.001:
		old_p_label.append('***')

ax.annotate(new_p_label[0], xy=(0,new_corr[2]),\
			horizontalalignment='left',verticalalignment='bottom')
ax.annotate(new_p_label[1], xy=(1,new_corr[3]),\
			horizontalalignment='left',verticalalignment='bottom')
ax.annotate(new_p_label[2], xy=(2,new_corr[4]),\
			horizontalalignment='left',verticalalignment='bottom')
ax.annotate(new_p_label[3], xy=(3,new_corr[5]),\
			horizontalalignment='left',verticalalignment='bottom')
# ax.plot(np.arange(6), new_p, color='red', label='newMAG p')
ax.scatter(np.arange(4), old_corr[2:], s=60, color='green', label='old MAG corr')
ax.annotate(old_p_label[0], xy=(0,old_corr[2]),\
			horizontalalignment='left',verticalalignment='bottom')
ax.annotate(old_p_label[1], xy=(1,old_corr[3]),\
			horizontalalignment='left',verticalalignment='bottom')
ax.annotate(old_p_label[2], xy=(2,old_corr[4]),\
			horizontalalignment='left',verticalalignment='bottom')
ax.annotate(old_p_label[3], xy=(3,old_corr[5]),\
			horizontalalignment='left',verticalalignment='bottom')
# ax.plot(np.arange(6), old_p, color='blue', label='oldMAG p')
plt.xticks(np.arange(4),['hum_len&#nodes','hum_len&#edges','opt_len&#nodes','opt_len&#edges'])
plt.legend(loc='upper right')
plt.suptitle('Spearman corr of MAG #nodes/#edge and human/optimal length')
plt.title('*: 0.01<p<=0.05, **: 0.001<p<=0.01, ***: p<=0.001',fontsize=7)
#plt.show()
plt.savefig(out_dir)
