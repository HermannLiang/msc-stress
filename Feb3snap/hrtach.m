function [nt,nv,nlabs] = hrtach(starttime,rrintervals,labels,sr,goodlabel)
%[nt,nv,nlabs] = hrtach(starttime,rrintervals,labels,sr,goodlabel)
% Contructs evenly sampled heart rate from rrintervals, using the
% Berger tachometer.
% <starttime> a scalar giving the time at which the first RR interval
% started.  If the RR intervals are calculated by differencing R-wave
% times, <starttime> is the first R-wave time.
% <rrintervals> is the rrinterval data.  These should be in seconds.
% <labels> tells whether each RR interval is valid or not.  If
% these are not provided, it is assumed that all beats are valid.
% <sr> the sampling rate of the tachometer (default: 4 samples/sec)
% <goodlabel> the numerical value of a valid label (default: 1).
% Returns:
% <nt> a vector of the time of each HR sample
% <nv> the HR data themselves
% <nlabs> labels for each data point.  1==valid, 0==invalid.
if nargin < 5
  goodlabel = 1;
end

if nargin < 4
  sr = 4; % samples per second
end

if nargin < 3
  labels = ones(size(rrintervals));
end

moment_length=1/sr; % in seconds 

  nt = zeros( ceil(sum(rrintervals)/moment_length) + 100, 1);
  nv = nt;
  nlabs = nt;
  outcount = 0;

%  HELP("A tachometer.  Reads several columns. The first column is a");
%  HELP("sequence of time intervals.  The remaining columns are the");
%  HELP("values of the signal during the corresponding time interval.");
%  HELP("The input intervals are assumed to be in seconds.");
%  HELP("The -hr flag is useful for calculating heart rate.");
%  IASSERT("-hr",hr_flag,"Add a column to the input that is 60/intervals.");
%  DCASE("-ml", moment_length, "How much time between output samples.(secs)");
%  IASSERT("-nt", no_times, "Don't put the times of the samples in the first column.");
%  PROCESS_COMS( argc, argv );

% A tachometer that turns rrintervals in evenly sampled HR.   
% This is based on the algorithm described in Ron Berger's IEEE Transactions
% on Biomedical Engineering Paper. 
%
% original C code: Daniel Kaplan, Dec. 28, 1989
% Copyright (c) 1989 by Daniel T. Kaplan
% translated to Matlab
% July 17, 1997
% Copyright (c) 1997 by Daniel T. Kaplan
% All Rights Reserved


% The idea here is to assume that we are given an input stream consisting
% of time intervals in the first column, and values in the second.  We
% want to make an evenly sampled output that represents what we would
% get if we assumed that the true analog signal is a series of steps.
% Each step has duration corresponding to the time interval, and amplitude
% corresponding to the value.  In order to produce the evenly sampled 
% signal, we want to anti-alias filter the hypothetical analog signal.
% We do this by integrating the analog signal over a boxcar.
%
% Actually, we will make the boxcar only half as wide as it needs to be.
% For the output, we will average the results of consecutive boxcars.

last_moment = 0.0;  % tells when the current boxcar ends
time_elapsed = 0.0; % Tells when the current step ends.  It is 
                    %          presumed that the step starts at the sample,
                    %          and ends after the time interval has gone by.


carry_overvals=0.0; % How much is left from the last 
                    % step to add to the integral of 
                    % the current boxcar. 
carry_overlabs=0.0; % label from the last step

time_to_fill=0.0;   % How much of the current boxcar covers the 
                    %          last step.  This is between 0 and 1 
how_much_already_filled=0.0; % How much of the current boxcar
                             %            has already been filled in by
                             %            previous steps. 
% The next two variables are used for averaging two consecutive boxcars   
previous_value=0;
previous_lab = 0; 
outvalue=0; 
not_initialized=1;  % A flag to tell us to fudge the averaging
                    %         of the first boxcar. 

% when does the current boxcar end?
last_moment = last_moment + moment_length;

% convert rr interval to hr units.
vals = 60./rrintervals;  % beats per minute
% convert labels to zero/one
labels = (labels == goodlabel);

% Note: the first value is ignored --- only the time stamp
for k=1:length(rrintervals)

    % when does the current step end? 
    time_elapsed = time_elapsed + rrintervals(k);

    % what is the amplitude of the current step : vals(k)

    % handle all the boxcars that fit into the current step. 
    while (last_moment <= time_elapsed) 
      % if the current boxcar ends during the current step, all we need
      % to do is figure out how much of the boxcar was filled during the
      % previous step (carry_over), and fill up the rest of it. */
      outvalue = vals(k)*(1.0 - how_much_already_filled) +carry_overvals;
      outlab = labels(k)*(1.0 - how_much_already_filled) +carry_overlabs;
      % outvalue is the integral of the signal over one boxcar.  But, 
      % the boxcars are only half as big as they need to be to prevent
      % aliasing.  Therefore, average together the contents of this 
      % boxcar and the previous one. 
      if not_initialized 
        % When dealing with the very first boxcar, remember that we
        %   don't have a previous value to average it with 
        previous_value = outvalue;
	previous_lab = outlab;
        not_initialized = 0;
      end
      % average two consecutive boxcars and write the output
      outcount = outcount+1;
      nv(outcount) = (outvalue + previous_value)/2.0;
      previous_value = outvalue;
      nt(outcount) = last_moment;
      nlabs(outcount) = (outlab+previous_lab)/2.0;
      previous_lab = outlab;

      % since the boxcar ended in this step, there will be nothing
      %  to carry over to the next boxcar 
      carry_overvals = 0.0;
      carry_overlabs = 0.0;
      how_much_already_filled = 0.0;
      time_to_fill = 0.0;
      % figure out when the next boxcar will end 
      last_moment = last_moment + moment_length;
    end

    % If the boxcar didn't end in the current step, there may be
    %   some remnant of the current step that should be carried over
    %   to the next boxcar.  Figure out what that is. 
    % If a step is completely contained within a boxcar, then we may be
    %   carrying over some from that step and some from the step or steps
    %   before.  how_much_already_filled tells how much of the current 
    %   boxcar was filled in previous steps.  
    time_to_fill = 1.0 - ( (last_moment - time_elapsed ) / moment_length);
    time_to_fill = time_to_fill - how_much_already_filled;
    carry_overvals = carry_overvals + vals(k)*time_to_fill;
    carry_overlabs = carry_overlabs + labels(k)*time_to_fill;
    how_much_already_filled = how_much_already_filled + time_to_fill;
  end


  nt = starttime + nt(1:outcount);
  nv = nv(1:outcount);
  nlabs = nlabs(1:outcount)>0.999;













