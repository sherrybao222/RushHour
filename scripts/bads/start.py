import matlab.engine, sys

if __name__ == '__main__':
	x0 = matlab.double([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
			0.01, 10, 0.01])
	lb = matlab.double([-5, -5, -5, -5, -5, -5, -5,
				0, 0, 0])
	ub = matlab.double([5, 5, 5, 5, 5, 5, 5,
				1, 50, 1])
	plb = matlab.double([-1, -1, -1, -1, -1, -1, -1,
				0, 1, 0])
	pub = matlab.double([5, 5, 5, 5, 5, 5, 5,
				0.5, 20, 0.5])


	eng = matlab.engine.start_matlab()
	sys.setrecursionlimit(31000)
	result = eng.bads("@ll", x0,lb,ub,plb,pub, nargout=2)
	print(result)
