function [t,v,lab] = apenhr(times,vals,labels,params)
% [t,v,lab] = apenhr(times,vals,labels,params)
% computes the approximate entropy of the data.
% <params> consists of [rstd lag]
% <rstd> gives the "filter factor" which is the length
% scale to apply in computing apen.  If rstd > 0,
% this is in terms of the standard deviation of the data.
% If rstd < 0, it's absolute value will be used as an
% absolute length scale.
% <lag> is the embedding lag.  
% Outputs:
% t   -- min and max times of the segment
% v   -- fraction of beat pairs differing by more than alpha
% lab -- fraction of beat pairs where one or both beats are invalid.
%
% When no arguments are given, the program documents itself
% [t,v,lab] = apenhr()
% <t> tells how many values in the returned t when there are arguments
% <v> tells the approximate entropy
% <lab> is a character string documenting t,v, and lab

% self documentation
if nargin < 3
  t = 2; % return the max and min in the segment
  v = 1; % return the pnnalpha
  % contents of the labels --- NOTE ALL MUST BE THE SAME LENGTH
  lab = [ 't1=min time      ';
          't2=max time      ';
          'v1=apen          ';
          'lb=frac bad beats'];
  return;
end



% set default parameters
if nargin < 4 
   rstd = 0.2;
   lag = 1;
else
  if length(params) ~= 2
    error('apenhr: Must give two params: [rstd, lag]');
  else
    rstd = params(1);
    lag = params(2);  
  end
end

% set the filter factor
if rstd > 0 
  goodindex = find(labels == 1);
  r = rstd*std(vals(goodindex));
else
  r = abs(rstd);
end

% embed the data in p=2
% and find the pre and post values
edata = lagembed(vals,2);
[pre,post] = getimage(edata,1);

% pull out only those rows that have good labels throughout
ldata = lagembed(labels,2,lag);
[lpre,lpost]= getimage(ldata,1);
foo = ~(any(~lpre')' | ~lpost);


npre = pre(find(foo));
npost = post(find(foo));


% actually do the calculation
v = apen(pre,post,r);
t = [min(times), max(times)];
lab = 1-(sum(foo)/length(foo));










