% Function for calculating multiscale entropy
% input: signal
% m: match point(s)
% r: matching tolerance
% factor: number of scale factor
% sampenc is available at http://people.ece.cornell.edu/land/PROJECTS/Complexity/sampenc.m

function e = msentropy(input,m,r,factor)

y=input;
y=y-mean(y);
y=y/std(y);

for i=1:factor
   s=coarsegraining(y,i);
   sampe=sampenc(s,m+1,r);
   e(i)=sampe(m+1);   
end
e=e';