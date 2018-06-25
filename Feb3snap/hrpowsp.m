function [t,v,lab] = hrpowsp(times, vals, labels, bands, detrend, goodlabel)
% [t,v,lab] = hrpowsp(times,vals,labels, bands, detrend, goodlabel)
% does power spectral analysis of a segment of data
% integrating up the energy in each of the specified bands.
% invalid beats are splined through
% NOTE: <bands> is specified in terms of HZ, and assumes
% that <times> is given in seconds.
% <goodlabel> identifies a valid sample (Default value: 1)
% <detrend> If > 0, subtract out a polynomial trend of the given order. 
%          (Default: 1, but the mean is always subtracted)
% Outputs:
% t -- min and max times of the segment
% v -- energies in the specified bands.  Units: energy/time-unit
% lab -- fraction of invalid beats

if nargin < 5
  detrend = 1;
end


if nargin < 6
  goodlabel = 1;
end

if nargin < 4
  bands = [0     0.003;
           0.003 0.04;
	   0.04  0.15;
	   0.15  0.4; ];
end

if nargin == 1  
  % gave only the bands as an argument,
  % but it will be called <times> in the argument list
  bands = times;
end

% When only the bands are given as an argument,
% or when there are no arguments, the
% program documents itself
% self documentation
if nargin <3 
  t = 2; % return the max and min in the segment
  [v,foo] = size(bands); % return the energy in each band
  % contents of the labels --- NOTE ALL MUST BE THE SAME LENGTH
  lab = [ 't1=min time      ';
          't2=max time      ';
          'v=energy in bands';
          'lb=frac bad data '];
  return;
end

%spline through glitches
[splinetimes,splinevals,splinelabels] = hrspline(times,vals,labels,goodlabel);

% detrend
if detrend > 0
  splinevals = hrtrend(splinetimes,splinevals,ceil(detrend) );
else
  if detrend == 0 
    splinevals = splinevals - mean(splinevals);
  end
end

%compute a representative sampling rate
%(which will be right if the points are evenly sampled)
maxtime = max(times);
mintime = min(times);

sr = (length(times)-1)/(maxtime-mintime);

% convert <bands> to be in units of the sampling
% frequency, instead of HZ
% bands = bands/sr;

v = bandstr(splinevals, sr, bands)'; %note transpose at end

% normalize to the length of the data segment
% v = v/(maxtime-mintime);
% this is already done by bandstr



t = [mintime,maxtime];
% make the label the fraction of bad beats
lab = 1-(sum(labels==1)/length(labels));
