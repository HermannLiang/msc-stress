function z = makerr(N)
% makerr(N)
% makes bogus RR interval data using the random number generator

% make some sinusoidal stuff
x = randn(N,1);
f1 = filtspec(.10,N,.01);
f2 = filtspec(.08,N,.01);
y = dfilt(x,f1-f2);

% add in a slow drift
x = randn(N,1);
xx = runsum(.1*x);

% add in a sine wave
y = y + xx + sin(xx);

% convert to units that are like RR intervals
y = .8 + .1*y; 
z = y;

