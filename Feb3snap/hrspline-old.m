function [ntimes,newvals,newlabels] = hrspline(times,vals,labs,goodlabel )
% [ntimes,newvals, newlabels] = hrspline(times, vals, labs, goodlabel )
% splines out bogus data by evaluating the rest of the data
% at the times at which the bogus data occurred.
% <labs> tells which are the bogus points, whichever are
% labeled with <goodlabel> (default: 1)

if nargin < 4
 goodlabel = 1;
end

newvals = vals;
newlabels = zeros(size(labs));

gs = find(labs == goodlabel);
bs = find(labs ~= goodlabel);

gtimes = times(gs);
gvals = vals(gs);
ntimes = times(bs);

nvals = interp1(gtimes, gvals, ntimes,'linear');

newvals(bs) = nvals;
newlabels(gs) = ones(size(gs));
ntimes = times;


% the interpolation may have put NaN at the endpoints --- since
% it refuses to extrapolate.  If so, replace these with the
% nearest valid point
if any( isnan(newvals) )
  nanindices = find( isnan(newvals) );
  goodindices = find( ~isnan(newvals) );
  newvals = newvals(:);
  ntimes = ntimes(:);
  nvals = interp1([-1000000000; ntimes(goodindices); 10000000000], [0; newvals(goodindices); 0], nanindices, 'nearest');
  newvals(nanindices) = nvals;
end