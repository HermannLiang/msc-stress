function resids = arhrdeg(data, inlabels, modelorder)
% res = hrdeglch(data, inlabels, modelorder, outcrit)
% identifies outliers in a time series by looking
% for outliers to a linear prediction model
% data -- the time series
% inlabels -- a zero for each already-known bogus data item
% modelorder -- what model order to use
% returns resids --- the residuals written in terms of IQRs from the median
%   these are synchronized to the data point being predicted.

% convert to a column format
data = data(:);
inlabels = inlabels(:);

if length(inlabels) ~= length(data)
  error('hrdeglch: data and labels must be the same length.');
end

% create a lag-embedded matrix of preimages
pre = ones( length(data) - modelorder, modelorder+1);
lpre = ones( length(data) - modelorder, modelorder+1);
% and the images of this
targs = data( (modelorder+1):(length(data)) );


% fill in the matrix
for k=1:modelorder
  foo = length(data) - modelorder;
  pre(:,(k+1)) = data( (modelorder+1-k):(length(data)-k) );
  lpre(:,(k+1)) = inlabels( (modelorder+1-k):(length(data)-k) );
end
lpre(:,1) = inlabels( (modelorder+1):(length(data)) );

% note that the matrix <pre> has all ones in the first column,
% to handle the constant part of the linear model
% the <lpre> matrix gives the labels of the correspond entries
% in <pre>
% get rid of the known bad beats
% by identifying any row with a bad beat
goodrows = find( all((lpre~= 0),2) );

% create matrices for the preimages and targets that have
% only clean beats
cleanpre = pre(goodrows,:);
cleantargs = targs(goodrows);

% fit the linear model
params = cleanpre\cleantargs;

% calculate the residuals
res = targs - pre*params;

% set the output to give the residual for the beat being
% predicted.  Make bogus residuals big so they
% are easily pulled out
resids = 100*ones(size(data));
resids( (modelorder + 1):(length(data)) ) = res;
% TO DO ON THIS
%
% 1) Write a supervisory script that runs this on the
% data both forward and backwards in time, and gives
% the minimum residual for each point.
%
% 2) Convert to IQR units.

