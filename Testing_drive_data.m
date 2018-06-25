%%
clear all
load('drive01m.mat');
data = val;
clear val;
fs = 15.5;
maxecg = max(data);
sdecg = std(data);
data = data/sdecg;
feats = fECGFeature2drive(data,fs);
%%
clear all

a = arduino;
