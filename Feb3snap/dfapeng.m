function [slow,fast] = dfapeng(vals,slowlen,midlen,fastlen)
% [slow,fast] = dfapeng(vals,slowlen,midlen,fastlen)
%
% Compute DFA in the manner described by C-K Peng,
% in C-K Peng, S Havlin, HE Stanley, AL Goldberger (1995)
% "Detrended Fluctuation Analysis" Chaos 5(1):82-87
% <vals> -- the data, typically RR intervals
% <slowlen> -- a slow time scale, 64 beats by default
% <midlen>  -- a medium time scale, 16 beats by default
% <fast>    -- a fast time scale, 4 beats by default
% Returned values:
% <slow>    -- the power-law slope on the slow to mid time scales
% <fast>    -- the power-law slope on the mid to fast time scales

if nargin < 4
  fastlen = 4;
end

if nargin < 3
  midlen = 16;
end

if nargin < 2
  slowlen = 64;
end

[num,resid] = dfamain( vals, fastlen, slowlen, sqrt(2) );

% fit the power-law of the faster time scales
goods = find( num >= fastlen & num <= midlen);
p = polyfit( log(num(goods)), log(resid(goods)), 1 );
fast = p(1);

% fit the power-law of the slower time scales
goods = find( num >= midlen & num <= slowlen);
p = polyfit( log(num(goods)), log(resid(goods)), 1 );
slow = p(1);








