function [t,v,lab] = pnnalpha(times, vals, labels, alpha)
% [t,v,lab] = pnnalpha(times,vals,labels,alpha)
% finds the fraction of consecutive beats that differ by
% more than a specified time.
% Setting <alpha> to 50 msec will make this equivalent to pnn50
% 50 msec is the default value of <alpha>
% Outputs
% t -- min and max times of the segment
% v -- fraction of beat pairs differing by more than alpha
% lab -- fraction of beat pairs where one or both beats are invalid.
%
% When no arguments are given, the program documents itself
% [t,v,lab] = pnnalpha()
% <t> tells how many values in the returned t when there are arguments
% <v> tells ...
% <lab> is a character string documenting t,v, and lab

% self documentation
if nargin < 3
  t = 2; % return the max and min in the segment
  v = 1; % return the pnnalpha
  % contents of the labels --- NOTE ALL MUST BE THE SAME LENGTH
  lab = [ 't1=min time      ';
          't2=max time      ';
          'v1=pnnalpha      ';
          'lb=frac bad beats'];
  return;
end



% set default parameters
if nargin < 4 
  % assume millisecs
 alpha = 50;
end

leng = length(vals);
%Make labels a column vector
labels = labels(:);
% create a matrix of labels
rrlabs = [labels(1:(leng-1)),labels(2:leng)];

% find which pairs have both beats valid
keepers = ~(any((~ (rrlabs')))');
% for Matlab 5
% keepers = ~any((~ (rrlabs)));


dif = diff(vals);

%keep the ones where both beats are good
dif = dif( find(keepers==1) );
% pnnalpha is the fraction of differences larger than alpha
v = sum( abs(dif) >= alpha )/length(dif);
t = [min(times),max(times)];
% make the label the fraction of bad beats
lab = 1-(sum(keepers)/length(keepers));

