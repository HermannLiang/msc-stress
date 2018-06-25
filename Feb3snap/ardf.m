function resid = Ardf(boxlength,intdata)

%	 resid = Ardf(boxlength,intdata)
%
%	 Ardf calculates the average rms of the detrended fluctuation 
%
%        intdata ... integrated input data
%
%        resid   ... result

N = length(intdata);
sumsq = 0;

for k = 0:boxlength:N-1
    if (k+boxlength > N)
       break
    end
    
    datausedcount = k+boxlength;
    ypoint = intdata(k+1:k+boxlength);

    ypoint = detrend(ypoint);
    sumsq = sumsq+sum(ypoint.^2);

end

resid = sqrt(sumsq/datausedcount);

