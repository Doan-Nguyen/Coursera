%               operations
% +, -, *, /, ^
%   ==, ~=, &&, ||, xor(a, b)

%               Variabels
% assign: a into 3
% a = 3; % semicolon supressing output.
% b = "hi";
% c = (1 <= 3)

%               print a variable
% disp(variable);
%   display a number with decimals
% disp(sprintf('2 decimals: %0.2f', a))


%               Vector & matrices
% matrices
% a = [1 2; 3 4; 5 6]
% vector
% b = [1 2 3 4] % (3, 1)
% c = [1; 2; 3; 4] % (1, 3)

%               Loop
% v = start_value:slide_value:end_value;
% v = start_value:end_value; % auto slide_value=1

%               Generate matrix
% ones(rows, cols)
% c = 2*ones(rows, cols)
%   random matrix with uniformly between 0 -> 1
% c = rand(rows, cols)
%   gaussian random vaiables (standard normal distribution)
% g = randn(rows, cols)

%               Visualization
% hist(input_matrix, numb_bins)  % split more data



%               Moving data around
% check size of matrix: size(A)
% >> get rows: size(A, 1)
% >> get cols: size(A, 2)
% get length of vector: length(v)
% if check length of matrix, the result is number of rows.

%               Load, find data on the file system.
% >>  load('input_file.dat');
% features_x = load('featuresX.dat');
% price_y = load('priceY.dat');

%               Create new data from old data
% v = price_y(1:10);
%   save to new data (*.dat)
% >> save new_data_path.dat v;
save hello.mat v;
%   load data:
% >> load new_data_path.dat;
load hello.mat;

%   get elements
% a(:, col_id);
% a(row_id, :);
% a([row_id_i row_id_j ...], :)

%   assign values
% a(:, col_id) = [row_1; row_2; ...];
% a(row_id, :) = [col_1; col_2; ...];

%   appdend another column vector to right 
% a = [a, [row_1; row_2; row_3, ...]]

%   put all elements of A into a single vector
% a(:)

%   concatenate 2 matrices vertical
% c = [a b];
%   concatenat2 matrices horizontal
% c = [a; b];