function[t, v, lab] = sdann(times, vals, labels, seglength)
% function[t, v, lab] = sdann(times, vals, labels, seglength)
% calculates:
% 1) the sdann: the standard deviation of the averages of values
% 2) the sdnn index:  mean of the standard deviations of all vals 
% for all segments
% over intervals of a specified constant length.
%
% <seglength> is the length of the interval requested 
% (it is defaulted at 5minutes == 300 seconds)
% NOTE: seglength must be specified in the same units as <times>, e.g. secs

if nargin < 4
	seglength==300;  % seconds
end

% self documentation
if nargin == 0
  t = 2; % return the [min,max] times
  v = 2; % return the sdann, and the sdnn index
  % contents of the labels --- NOTE ALL MUST BE THE SAME LENGTH
  lab = [ 't1=min time      ';
	  't2=max time      ';
          'v1=sdann         ';
          'v2=sdnnindex     ';
          'lb=frac bad beats'];
  return;
end

startends=makesegs(times, seglength);
[ts,valu,lbs]=hrsegment(times, vals, labels, startends, 'simpstat');
means=valu(:,1);
stdev=valu(:,2);
sda=std(means);
sdi=mean(stdev);
v=[sda,sdi];
lab=lbs;
t=[min(times),max(times)];
