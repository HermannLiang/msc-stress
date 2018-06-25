function [t,v,lab] = hrdfa(times,vals,labels,params)
% [t,v,lab] = hrdfa(times,vals,labels,params)
% A member of the family of statistics for use in the HRV system
% This one calculates the DFA in the manner of CK Peng.
% returning a time entry, a value (containing the slow and fast
% power law exponents), and a label 
% (reflecting the fraction of bad beats in the
% segment.
%
% labels should be 1 for good beats, 0 for bad beats.
% params is not used?

% self documentation
if nargin < 3 
  t = 2; % return the max and min in the segment
  v = 2; % return the fast and slow dfa power-law exponents
  % contents of the labels --- NOTE ALL MUST BE THE SAME LENGTH
  lab = [ 't1=min time      ';
          't2=max time      ';
          'v1=fast power exp';
          'v2=slow power exp';
          'lb=frac bad beats'];
  return;
end

[slow,fast] = dfapeng(vals,64,16,4);
v = zeros(1,2);
v(1) = fast;
v(2) = slow;

% fraction of vals that are invalid
lab = sum(labels)/length(labels);

t = [ min(times), max(times) ];






