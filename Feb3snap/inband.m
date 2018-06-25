function s = inband(powspec,firstbin,lastbin)
% s = inband(powspec,firstbin,lastbin)
% calculates the integral between first and last
% powspec -- the left half on an fft
% firstbin -- the lower index (not frequency, but bin number)
% lastbin  -- the upper index (not frequency, but bin number)

foo = 1:length(powspec);


gbins=find(foo >= firstbin & foo <= lastbin);
goo=powspec(gbins);
s = sum(goo);
