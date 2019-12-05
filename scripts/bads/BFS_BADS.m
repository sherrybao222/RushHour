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

x0 = [-0.5391 -3.9844 3.8281 -5.5469 -2.5781 -8.8867 -2.5469 -0.7656 1.4844 5.3320];                 % Starting point
lb = [-2 -20 -20 -20 -20 -20 -20 -20 -15 0];             % Lower bounds
ub = [2 10 10 10 10 10 10 10 15 20];               % Upper bounds
plb = [-1 -15 -15 -15 -15 -10 -10 -10 -10 0];              % Plausible lower bounds
pub = [1 5 5 5 5 5 2 2 10 15];                % Plausible upper bounds

% Screen display
fprintf('\n');
display('*** Example 1: Basic usage');
display('  Simple usage of BADS on <a href="https://en.wikipedia.org/wiki/Rosenbrock_function">Rosenbrock''s banana function</a> in 2D.');
display('  Press any key to continue.'); fprintf('\n');
% pause;
display('  Continued.');

% Run BADS, which returns the minimum X and its value FVAL.
[x,fval] = bads(@ll,x0,lb,ub,plb,pub)

% Note that BADS by default does not aim for extreme numerical precision 
% (e.g., beyond the 2nd or 3rd decimal place), since in realistic 
% model-fitting problems such a resolution is typically pointless.

