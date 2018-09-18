%%
A = [1 2 10; 3 4 20; 9 6 15];
C = bsxfun(@minus, A, mean(A));