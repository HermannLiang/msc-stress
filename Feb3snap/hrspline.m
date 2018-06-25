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

if length(gs) == 0 
  warning('hrspline: no valid data in a segment.  Returning zeros');
  ntimes = times;
  newvals = zeros(size(vals));
  newlabs = zeros(size(vals));
  return;
end


gtimes = times(gs);
gvals = vals(gs);
ntimes = times(bs);
inside = find( ntimes >= min(gtimes) & ntimes <= max(gtimes) );
nvals = NaN.*ones(size(ntimes));

%nvals = interp1(gtimes, gvals, ntimes,'linear');
%if (inside ~= []) 

if ~isempty(inside)
        nvals(inside) = interp1(gtimes,gvals, ntimes(inside), 'linear');
end;

newvals(bs) = nvals;
newlabels(gs) = ones(size(gs));
ntimes = times;


% the interpolation may have put NaN at the endpoints --- since
% it refuses to extrapolate.  If so, replace these with the
% nearest valid point
if any( isnan(newvals) )
  nanindices = find( isnan(newvals) );
  goodindices = find( ~isnan(newvals) );
  nanleft = nanindices(find(nanindices < goodindices(1)));
  nanright = nanindices(find(nanindices > goodindices(length(goodindices))));
  if (length(nanleft) + length(nanright) < length(nanindices))
        %This should never happen
        error('Non-interpolated points present, not at endpoints.');
  else
        for i = nanleft'
                newvals(i) = newvals(goodindices(1));
        end;
        for i = nanright'
                newvals(i) = newvals(goodindices(length(goodindices)));
        end;
  end;
end


