from BFS import *
import pandas as pd
from statsmodels.base.model import GenericLikelihoodModel

def test_all_legal_moves(car_list, answer):
	all_moves = all_legal_moves(car_list, Board(car_list))
	assert len(all_moves) == answer, 'test_all_legal_moves FAILED, observe '+str(len(all_moves))

def test_InitializeChildren(node, params, answer):
	InitializeChildren(node, params)
	assert len(node.children) == answer, 'test_InitializedChildren FAILED, observe '+str(len(node.children))
	test_Node(node.children[1], params)

def test_move(car_list, car_tag, to_position, params, answer):
	print('-------------- test move ----------------')
	new_list, new_red = move(car_list, car_tag, to_position)
	new_node = Node(new_list, params)
	test_Node(new_node, params)
	test_InitializeChildren(new_node, params, answer)
	test_all_legal_moves(new_list, answer)

def test_red(node):
	assert node.red == node.board.red, 'test_red FAILED'

def test_Node(node, params):
	print('--------------- test node ---------------')
	print(node.board_to_str())
	test_red(node)
	print(node.value)
	for car in node.car_list:
		print('Car '+ car.tag 
			+ ', edge_to:'+str([str(i.tag) for i in car.edge_to])
			+ ', levels:'+str(car.level))

def test_is_solved(board, red, answer):
	assert is_solved(board, red) == answer, "test_is_solved FAILED"

def my_negll(x,y,beta,sigma=1):
	'''
		return negative log likelihood for current parameters
		BADS will find minimum based on this function value
	'''
	N = len(y)
	RSS = np.sum((y-(beta[0]*x[:,-1]+beta[1]))**2) #residual sum of squares
	return np.log(np.sqrt(2*np.pi*sigma**2))+(1/(2*sigma**2))*RSS

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
    return slope, intercept

class MyOLS(GenericLikelihoodModel):
	def __init__(self, enog, exog, **kwds): # endog=y, exog=x
		super(MyOLS, self).__init__(enog, exog, **kwds)
	def nloglikeobs(self, params):
		sigma = params[-1]
		beta = params[:-1]
		# print('params: '+str(params))
		# print('sigma: '+str(sigma))
		# print('beta: '+str(beta))
		return my_negll(self.exog, self.endog, beta, sigma)
	def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
		# we have one additional parameter and we need to add it for summary
		self.exog_names.append('sigma')
		if start_params == None:
		    # Reasonable starting values
		    start_params = np.append(np.zeros(self.exog.shape[1]), .5)
		    print('start_params '+str(start_params))
		return super(MyOLS, self).fit(start_params=start_params,
					     maxiter=maxiter, maxfun=maxfun,
					     **kwds)
def test_MLE():
	N = 10
	x = 10 + 2*np.random.randn(N)
	y = 5 + x + np.random.randn(N)
	df = pd.DataFrame({'y':y, 'x':x})
	df['constant'] = 1

	print('x: '+str(x))
	print('y: '+str(y))

	sm_ols_manual = MyOLS(df.y,df[['constant','x']]).fit()
	print(sm_ols_manual.summary())
	print('find_line: '+str(find_line(x,y)))



