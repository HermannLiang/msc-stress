function [ features ] = fRESFeature2( data,fs )
%fGSRFeature2 Summary of this function goes here
%   Detailed explanation goes here
%   Extract the feaature from respiration data
%   --------------------------Inputs---------------------------------------
%   data - M x N matrix. There are M observations and N dimensions.
%   num_channel - Number of channels.
%   --------------------------Outputs--------------------------------------
%   features - M x (? x num_channel)
%   -------------------------Other Info------------------------------------
%   Date: 14 JUNE 2016
%   Programmer: Youqian ZHANG 
%   Organization: Imperial College London 
%   Contact: youqian2012@gmail.com
%   ---------------------------End-----------------------------------------

[M,len] = size(data); % Number of obeservations
features = zeros(M,17);

for i = 1:M
    sig = RES_aqn_variable(data(i,:), fs);
    [RES_feats, ~] = RES_feat_extr(sig);
    MSE = multiScaleEntropy(data(i,:),5);
    features(i,:)=[RES_feats,MSE];
end
% features =zeros(40,1);
end

