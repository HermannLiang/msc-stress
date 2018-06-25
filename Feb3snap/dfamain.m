function [num, resid] = dfa(data, smallestbox, largestbox, step)

%	 [num, resid] = dfa(data, smallestbox, largestbox, step)
%
%	 dfa finds the detrended fluctuations
%
%        data        ... input data
%        smallestbox ... smallest window size to use
%        largestbox  ... largest window size to use
%        step        ... stepsize

dcount = 0;
N = 0;
num = 0;

if nargin < 2
   smallestbox = 4;
   largestbox = 0;
   step = 1.4142136;
end

N = length(data);

idata = dfahelpr(data);

if largestbox == 0
   largestbox = N;
elseif largestbox < 0
   largestbox = N/4;
end

i = smallestbox;
j = 1;

while floor(i) <= largestbox

% Danny replaced these lines, so that the log is not taken
%  resid(j) = log10(Ardf(floor(i),idata));
%  num(j) = log10(floor(i));
  resid(j) = (ardf(floor(i),idata));
  num(j) = (floor(i));
  i = i*step;
  j = j+1;

end








