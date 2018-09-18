clear all
clc
s = serial ('COM3');
set(s,'BaudRate',115200);
if ~isempty(instrfind)
    fclose(instrfind);
    delete(instrfind);
end
fopen(s);
fprintf(s,'*IDN?')
out = fscanf(s);
%%
clear 
M1 = dlmread('justatest4.txt');
post = M1(:,1) >0;
plot(M1(post,1));
% M1(:,1) = M1(:,1)./1000000;
% M1(:,1) = 1./M1(:,1);
%%
clear all
close all
M1 = dlmread('justatest10.txt');
con = 1e6./M1(:,2);
post = M1(:,1) >0;
tstart = M1(1,4); tend = M1(end,4);
taxis = linspace(0,(tend-tstart)/1e3,length(M1));
figure,
plot(taxis,con); 
vol = M1(:,1);
figure,
plot(vol(post));
%%
clear 
M1 = dlmread('test02.txt');
plot(M1(:,1));

