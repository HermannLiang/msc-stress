function [t,v,labs] = hrsegment(times, vals, labels, startends, stat, params)
% [t,v,labs] = segment(times, vals, labels, startends, stat, params)
% Applies statistic <stat> to segments of a data set <times,vals,labels>
% <startends> is a matrix with two columns, giving the starting and
% ending times of each segment.
% <params> gets handed off to <stat> as an argument.
% <stat> is a quoted string containing the name of the function to use
% e.g., 'stdseg'

if nargin < 6
 params = [];
end

[a,b] = size(startends);
if b ~= 2
  error( 'segment: <startends> should be a 2 column matrix');
end

if a < 1
  error( 'segment: no entries given in <startends>');
end


% Create the output matrices
% How big should they be?

if nargin >= 6
  [twidth,vwidth,labsnames] = eval( [stat,'(params)'] );
else
  % if params is not given on the argument line, don't pass it as an argument
  [twidth,vwidth,labsnames] = eval( stat );
end

t = zeros(a,twidth);
v = zeros(a,vwidth);
labs = zeros(a,1);

%loop over the segments
for k=1:a
  first = startends(k,1);
  last  = startends(k,2);
  indices = find( (times >= first) & (times <= last) );
  tt = times(indices);
  vv = vals(indices);
  llabels = labels(indices);
  if nargin >= 6
    [tr,vr,lr] = eval([stat, '(tt,vv,llabels,params)']);
  else
    % params wasn't specified
    [tr,vr,lr] = eval([stat, '(tt,vv,llabels)']);
  end
  t(k,:) = tr;
  v(k,:) = vr;
  labs(k,:) = lr;
end




