function negLL = MLERegression(params)
    negLL = python('test.py', params[0], params[1], params[2])
end