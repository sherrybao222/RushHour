%BADS_EXAMPLES Examples for Bayesian Adaptive Direct Search

display('Running a number of examples usage for Bayesian Adaptive Direct Search (BADS).');
display('Open ''bads_examples.m'' to see additional comments and instructions.');

%% Example 2: Basic usage

% Simple usage of BADS on Rosenbrock's banana function in 2D
% (see https://en.wikipedia.org/wiki/Rosenbrock_function).
% 
% We specify wide hard bounds and tighter plausible bounds that (hopefully) 
% contain the solution. Plausible bounds represent your best guess at 
% bounding the region where the solution might lie.

x0 = [1 1];                 % Starting point
lb = [-10 -10];             % Lower bounds
ub = [10 10];               % Upper bounds
plb = [-5 -5];              % Plausible lower bounds
pub = [5 5];                % Plausible upper bounds

% Run BADS, which returns the minimum X and its value FVAL.

[x,fval] = bads(@test_ll,x0,lb,ub,plb,pub)

% Note that BADS by default does not aim for extreme numerical precision 
% (e.g., beyond the 2nd or 3rd decimal place), since in realistic 
% model-fitting problems such a resolution is typically pointless.

