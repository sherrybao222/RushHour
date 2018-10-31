import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

new_corr = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_corr.npy')
new_p = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_p.npy')
old_corr = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_old_corr.npy')
old_p = np.load('/Users/chloe/Documents/RushHour/data/len_MAG_old_p.npy')
out_dir = '/Users/chloe/Documents/RushHour/figures/corr_p.png'

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(6), new_corr, color='orange', label='newMAG corr')
ax.plot(np.arange(6), new_p, color='red', label='newMAG p')
ax.plot(np.arange(6), old_corr, color='green', label='oldMAG corr')
ax.plot(np.arange(6), old_p, color='blue', label='oldMAG p')
ax.set_xticklabels(['hum&#n+#e','opt&#n+#e','hum&#n','hum&#e','opt&#n','opt&#e'])
plt.legend(loc='upper right')
plt.title('summary of #success(first trial) #restart #surender')
#plt.show()
plt.savefig(out_dir)
