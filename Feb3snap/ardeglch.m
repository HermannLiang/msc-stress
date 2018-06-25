function [labs,resids] = ardeglch(data, inlabels, modelorder, iqrcrit)
% labs = arresid(data, inlabels, modelorder, iqrcrit)
% identifies outliers in a time series by
% looking at the residuals of a forward and
% backward AR fit.
% Excludes from the fit beats labeled 0 in <inlabels>
% iqrcrit gives the criteria in inter-quartile range units
% for an outlier.
% Returns a vector of the same length as the data which
% contains a 0 for any beat marked as bad either in inlabels
% or from the AR fit.
% [labs,resids] = ardeglch...
% gives the actual values of the residuals as an optional second argument

if nargin < 3 
  modelorder = 3;
end

if nargin < 4
  iqrcrit = 3.5;
end

% fit the forward model
labforward = arresid(data, inlabels, modelorder);


% fit the backward model
labbackward = arresid( data((length(data)):-1:1), inlabels((length(data)):-1:1), modelorder);


% take the smaller of the two residuals, remembering
% to put labbackward back in forward order.
labels = min(labforward, labbackward(length(data):-1:1) );
resids = labels;
% Compute the interquartile range and limits for outliers.
lims = prctile(labels,[25 50 75]);
iqrange = lims(3) - lims(1);
bottom = lims(1) - iqrange*iqrcrit;
top = lims(3) + iqrange*iqrcrit;

% bogus points are marked as 666666 in <labels> or as 0 in <inlabels>
labs = (labels > bottom & labels < top & labels ~= 666666 & inlabels ~= 0 );


