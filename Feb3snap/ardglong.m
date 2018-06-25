function reslabels = ardglong(data,inlabels,seglength,modelorder,iqrcrit)
% reslabels = ardglong(data,inlabels,seglength,modelorder,iqrcrit)
% Use AR modeling to deglitch a long time series by
% breaking it up segments, deglitching separately, and then
% concatenating the labels of the segments

% minseglength is the smallest segment we would want to process
minseglength = 30;

% set default values
if nargin < 5 
  iqrcrit = 3.5;
end

if nargin < 4
  modelorder = 3;
end

if nargin < 3
  seglength=300;
end

reslabels = 0*inlabels;

for first=1:seglength:(length(data)-minseglength);
  last = min(length(data),first+seglength-1);
  % if we are going to leave a short data segment at the end,
  % include it in the next to the last segment (which will then be the last)
  if ((length(data)-last) < minseglength )
    last = length(data);
  end

  % grab the data and labels
  indices = (first:last)';
  res = ardeglch(data(indices), inlabels(indices), modelorder, iqrcrit );
  reslabels(indices) = res;  
end









