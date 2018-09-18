% Function for calculating sample entropy
% input: signal
% m: match point(s)
% r: matching tolerance
% sampenc is available at http://people.ece.cornell.edu/land/PROJECTS/Complexity/sampenc.m

function e = spen(input,m,r)

y=input;
y=y-mean(y);
y=y/std(y);
sampe=sampenc(y,m+1,r);
e=sampe(m+1);   
