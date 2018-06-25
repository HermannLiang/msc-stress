function res = dodfa( name )
% do the dfa analysis for a file from the disk
data = readfile(name);

times = runsum(data)/1000; % convert to seconds from millisecs
labels = ones(size(times));
startends = makesegs(times,8000,0);

[t,v,labs] = hrsegment(times,data,labels,startends,'hrdfa');
res = v;