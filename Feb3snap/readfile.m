function res = readfile( name )
% res = readfile(name)
% read in the data from the named file, and return it
% as a variable.
load(name);
dot = find(name=='.');
varname = name(1:((dot(1)-1)));
res = eval(varname);
