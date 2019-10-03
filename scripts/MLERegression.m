function negLL = MLERegression(param0, param1, param2)
yhat = param0 + param1*df.x;
pd = makedist('Normal','mu',yhat,'sigma',param2);
negLL = -sum(log(pdf(pd, df.y));
end