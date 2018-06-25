function bin = findbin(N, rate, freq)
% bin = findbin(N, rate, freq)
% find the bin in an FFT corresponding to a given
% frequency
% N is the length of the data
% the sampling rate (samples/sec) of the data
% the desired frequency whose bin we want to know

bin = 1+ (freq*N/rate);