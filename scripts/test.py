# import libraries
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
import scipy.stats as stats
import sys
import matlab.engine
import scipy.io as sio

# generate data
N = 3
x = np.linspace(0,20,N)
e = np.random.normal(loc = 0.0, scale = 5.0, size = N)
y = 3*x + e
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 1


# # split features and target
# X = df[['constant', 'x']]
# # fit model and summarize
# sm.OLS(y,X).fit().summary()


# define likelihood function
def MLERegressionPy(param0, param1, param2):
	intercept, beta, sd = param0, param1, param2 # inputs are guesses at our parameters
	yhat = intercept + beta*x # predictions
# next, we flip the Bayesian question
# compute PDF of observed values normally distributed around mean (yhat)
# with a standard deviation of sd
	negLL = -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) )
# return negative LL
	return(negLL)


# define likelihood function
def MLERegression(params):
	intercept, beta, sd = params[0], params[1], params[2] # inputs are guesses at our parameters
	yhat = intercept + beta*x # predictions
# next, we flip the Bayesian question
# compute PDF of observed values normally distributed around mean (yhat)
# with a standard deviation of sd
	print(params)
	negLL = -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) )
# return negative LL
	return(negLL)

guess = np.array([5,5,2])
results = minimize(MLERegression, guess, method = 'Nelder-Mead', options={'disp': True})
print(results)

# if __name__ == '__main__':
# 	param0, param1, param2 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
# 	sys.stdout.write(str(MLERegressionPy(param0, param1, param2)))