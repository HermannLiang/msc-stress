function [t,v,lab] = simpstat(times,vals,labels,params)
% [t,v,lab] = simpstat(times,vals,labels,params)
% A member of the family of statistics for use in the HRV system
% This one calculates the mean and standard deviation
% returning a time entry, a value (containing the mean and standard
% deviation), and a label (reflecting the fraction of bad beats in the
% segment.
%
% labels should be 1 for good beats, 0 for bad beats.
% params is not used?

% self documentation
if nargin < 3 
  t = 2; % return the max and min in the segment
  v = 3; % return the mean, std dev and skewness of the data
  % contents of the labels --- NOTE ALL MUST BE THE SAME LENGTH
  lab = [ 't1=min time      ';
          't2=max time      ';
          'v1=mean          ';
          'v2=std deviation ';
          'v3=skewness      ';
          'lb=frac bad beats'];
  return;
end



goods = find(labels == 1);
v = zeros(1,3);
v(1) = mean(vals(goods));
v(2) = std(vals(goods));
v(3) = mean((vals(goods) - v(1)).^3)/(v(2)^3);
% fraction of vals that are invalid
lab = 1- length(goods)/length(vals);

t = [ min(times), max(times) ];



