A = [1 2; 3 4; 5 6];
B = [11 12; 13 14; 15 16];
C = [1 1; 2 2];
%           
mul_AC = A*C;
%   element-wise operations
ewo_AB = A.*B;
%    squared
sqr_A = A.^2;
%    minus
m_1 = 1 ./ A;
%    log 
log_a = log(A);
%    exponentialtion ~ lua thua
exp_a = exp(A);
%    abs
T = [-1 2 -5];
abs_a = abs(T);
%    negative
neg_a = -A;
%    transpose matrix
trans_a = A';
A = (A')';
%   get max/min value in matrix 
max_val = max(A);
%   get max/min value & index from vector
row1_a = A(1, :);
row1_a
[min_val, idx] = min(row1_a);
%   find element with conditions
A
cond_a = find(A < 3);
cond_a
