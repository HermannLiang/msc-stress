function energies = bandstr(vals, rate, bands);
% energies = bandstr(vals, rate,bands);
% calculates the amount of energy in a band of the power spectrum.
% vals  -- the data
% bands -- a vector; each row is low freq, high freq.
% rate  -- the sampling rate in hertz

%take power spectrum store in bins and val;
powspec = abs(fft(vals)).^2;

%normalize to length
powspec = powspec/(length(vals)^2);

% since we want to consider only positive frequencies,
% we need to double the values of everything except DC and Nyquest

% is it even or odd in length
% (don't use mod -- it's not in Matlab)
iseven = ~rem(length(vals),2);

% double everything
pspec = 2*powspec;

% set DC back to undoubled value
pspec(1) = powspec(1);
if iseven 
  % Nyquist back to undoubled values
  nybin = 1+length(vals)/2;
  pspec(nybin) = powspec(nybin);
end



nyq=rate/2; % for even time series
%((length-1)/2)/(length/rate) for odd time series


% loop over the bands

[bandn,goo] = size(bands);
if goo ~= 2
 error('bands must be ......');
end

energies = zeros(bandn,1);

for k=1:bandn
  lowfreq = bands(k,1); 
  highfreq= bands(k,2);
  N=length(vals); 
  %test for if freq are ok
  lowbin  = findbin(N, rate, lowfreq);
  highbin = findbin(N, rate, highfreq);

  energies(k) = inband(pspec,lowbin,highbin);
end



