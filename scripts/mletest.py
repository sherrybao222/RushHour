from scipy.optimize import minimize
import pandas as pd
import numpy as np
import scipy.stats as stats

def find_line(xs, ys):
    """Calculates the slope and intercept"""
    # number of points
    n = len(xs)
    # calculate means
    x_bar = sum(xs)/n
    y_bar = sum(ys)/n   
    # calculate slope
    num = 0
    denom = 0
    for i in range(n):
        num += (xs[i]-x_bar)*(ys[i]-y_bar)
        denom += (xs[i]-x_bar)**2
    slope = num/denom
    # calculate intercept
    intercept = y_bar - slope*x_bar
    print('slope: '+str(slope)+', intercept: '+str(intercept))
    return slope, intercept


def MLERegression(params):
	intercept, beta, sd = params[0], params[1], params[2] # inputs are guesses at our parameters
	yhat = intercept + beta*x # predictions
	# next, we flip the Bayesian question
	# compute PDF of observed values normally distributed around mean (yhat)
	# with a standard deviation of sd
	negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd) )
	# return negative LL
	return(negLL)

N = 100
x = np.linspace(0,20,N)
ϵ = np.random.normal(loc = 0.0, scale = 5.0, size = N)
y = 3*x + ϵ
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 1

guess = np.array([5,5,2])
results = minimize(MLERegression, guess, method = 'Nelder-Mead',
					options={'disp': True})
print(results)

find_line(x,y)






