function startends = makesegs(times,seglength,overlap,nsegs,startskip)
% startends = makesegs(times,seglength,segskip)
% create a list of start and end times for use in 
% <segment>.  This makes all segments the same length,
% and they can overlap (or if overlap is negative, skip)
% if <nsegs> is specified, only the first nsegs segments will be created
% <startskip> skips the first part of the data

if nargin < 5
 startskip = 0;
end

if nargin < 4
 % keep all the segments
 nsegs = -1; 
end

if nargin < 3 
 overlap = 0;
end

first=min(times)+startskip;
last = max(times)-seglength;

starts = (first:(seglength-overlap):last)';
ends = starts + seglength;

startends=[starts,ends];

if length(startends) == 0
  startends = [min(times)+startskip, max(times)];
  warning('Data too short for a full-length segment.  Creating a short segment.')
end


if nsegs >= 0
  startends = startends(1:(min(length(starts),nsegs)),:);
end


