function y = ll(x)
%ROSENBROCKS Rosenbrock's 'banana' function in any dimension.
% a = [x(1) x(2) x(3) x(4) x(5) x(6) x(7) x(8) x(9) x(10)];
% assignin('caller','a', a);
% y = py.BFS_ibs.ibs_interface();
y = py.BFS_ibs.ibs_interface(x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8),x(9),x(10));
disp y;