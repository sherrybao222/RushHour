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

x0 = [0.5391 3.9844 3.8281 5.5469 2.5781 8.8867 2.5469 0.05];                 % Starting point
lb = [-20 -20 -20 -20 -20 -20 -20 0];             % Lower bounds
ub = [10 10 10 10 10 10 10 1];               % Upper bounds
plb = [-15 -15 -15 -15 -10 -10 -10 0];              % Plausible lower bounds
pub = [5 5 5 5 5 2 2 1];                % Plausible upper bounds

% Run BADS, which returns the minimum X and its value FVAL.

[x,fval] = bads(@ll,x0,lb,ub,plb,pub)

% Note that BADS by default does not aim for extreme numerical precision 
% (e.g., beyond the 2nd or 3rd decimal place), since in realistic 
% model-fitting problems such a resolution is typically pointless.

