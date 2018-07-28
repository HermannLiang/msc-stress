clear
clc
close all
% examine the features, with visualisation enable
subjectno = 4;  %no. of subject
vis = 1;
para.T_winl = 250;
para.T_incre = 10;
para.F_winl = 250;
para.F_incre = 10;
para.F_pwinl = 150;
para.F_over = 10;
para.N_winl = 250;
para.N_incre = 10;
% input subject number, output features.
feat = getfeatures(subjectno,para,vis);

%standardizing: remove the median values from rest stage ??
%
%Selecting features: analysis of variance, p-test, anova, GA, 
%construct the machine learning network

%%
sequence
