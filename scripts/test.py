# BFS model

# import MAG, BFS
# import random, sys, copy
# import numpy as np
# import pymc3 as py

# basic_model = py.Model()

# with basic_model:

# 	w0 = py.Normal('w0', mu=1.0, sigma=1.0)
# 	w1 = py.Normal('w1', mu=1.0, sigma=1.0)
# 	w2 = py.Normal('w2', mu=1.0, sigma=1.0)
# 	w3 = py.Normal('w3', mu=1.0, sigma=1.0)
# 	w4 = py.Normal('w4', mu=1.0, sigma=1.0)
# 	w5 = py.Normal('w5', mu=1.0, sigma=1.0)
# 	w6 = py.Normal('w6', mu=1.0, sigma=1.0)
# 	w7 = py.Normal('w7', mu=1.0, sigma=1.0)
# 	noise = py.Normal('noise', mu=1.0, sigma=1.0)
	
# 	freq = BFS.Main(w0, w1, w2, w3, w4, w5, w6, w7, noise)
# 	print(freq)

from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import BFS

# data = sm.datasets.spector.load_pandas()
# exog = data.exog # dependent var
# endog = data.endog # regressors
vlevel, _, _ = BFS.Main()
exog = np.expand_dims(np.array(vlevel), axis=0)
endog = np.ones((1,len(vlevel)))


# print(sm.datasets.spector.NOTE)
# print(data.exog.head())

exog = sm.add_constant(exog, prepend=False)

class MyProbit(GenericLikelihoodModel):
    def loglike(self, params): 
    	print(params)
    	_, likelihood, _ = BFS.Main(params[0], params[1], \
    									params[3], params[7], params[9], params[11],\
    									params[13])
        return np.sum(np.log(likelihood))

sm_probit_manual = MyProbit(endog, exog).fit()
print(sm_probit_manual.summary())

print(sm_probit_manual.params)

