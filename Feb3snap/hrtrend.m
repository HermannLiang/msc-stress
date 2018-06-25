function [newvals,trend] = hrtrend(times,vals,polyorder)
% [newvals,trend] = hrtrend(times,vals,polyorder)
% detrends <vals> versus <time>  using a polynomial of <polyorder>

if polyorder < 1
  warning('hrtrend: polyorder must be integer > 0 ');
  newvals = vals;
  trend = 0*vals;
  return;
end

if polyorder > (length(vals) - 1)
  warning('hrtrend: polyorder is too long.  Must be < length(vals).  Resetting.');
  trend = vals;
  newvals = 0;
  return;
end

p = polyfit(times,vals,polyorder);
trend = polyval(p,times);
newvals = vals - trend;
