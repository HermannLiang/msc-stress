function [t,v,lab]=rmssd(times, vals, labels);
% function[t,v,labs]=rmssd(times, vals, labels);
% calculates the rmssd: the square root of the mean 
% of the sum of the squares of the
% differences.
% standard format.
% the label returned is the fraction of bogus beat pairs in the sample.
% the t returned is the [min,max] of the time range in the data
% the v returned is the value for the statistic.

if nargin == 0
  t = 2; % return the max and min in the segment
  v = 1; % return the rmssd
  % contents of the labels --- NOTE ALL MUST BE THE SAME LENGTH
  lab = [ 't1=min time      ';
          't2=max time      ';
          'v1=rmssd         ';
          'lb=frac bad beats'];
  return;
end

leng = length(vals);
%Make labels a column vector
%labels = labels(:);

% create a matrix of labels
rrlabs = [labels(1:(leng-1)),labels(2:leng)];

% find which pairs have both beats valid
keepers = ~(any((~(rrlabs')))');

dif = diff(vals);

%keep the ones where both beats are good
dif = dif( find(keepers==1) );

v=sqrt(mean(dif.*dif));

t=[min(times),max(times)];
% label is the fraction of bad beats
lab = 1-(sum(keepers)/length(keepers));