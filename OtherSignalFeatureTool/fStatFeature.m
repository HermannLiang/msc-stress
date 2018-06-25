function [ features ] = fStatFeature( data,num_channel )
%fStatFeature Summary of this function goes here
%   Detailed explanation goes here
%   Extract the statistical features.
%   --------------------------Inputs---------------------------------------
%   data - M x N matrix. There are M observations and N dimensions.
%   num_channel - Number of channels.
%   --------------------------Outputs--------------------------------------
%   features - M x (6 x num_channel)
%   -------------------------Other Info------------------------------------
%   Date: 14 JUNE 2016
%   Programmer: Youqian ZHANG 
%   Organization: Imperial College London 
%   Contact: youqian2012@gmail.com
%   ---------------------------End-----------------------------------------

M = size(data,1);
num_gp = M./num_channel;
features = zeros(num_gp,8.*num_channel);

%% Raw data
% 1. Mean of raw data
f1 = mean(data,2);
% 2. Var of raw data
f2 = var(data,0,2);
% 3. Mean of absolute of first difference of raw data
f3 = mean(abs(diff(data,1,2)),2);
% 4. Mean of absolute of second difference of raw data
f4 = mean([abs(diff(data(:,1:2:end),1,2)),abs(diff(data(:,2:2:end),1,2))],2);

%% Normalized data
data = normr(data);
% 5. Mean of raw data
f5 = mean(data,2);
% 6. Var of raw data
f6 = var(data,0,2);
% 7. Mean of absolute of first difference of normalized data
f7 = mean(abs(diff(data,1,2)),2);
% 8. Mean of absolute of second difference of normalized data
f8 = mean([abs(diff(data(:,1:2:end),1,2)),abs(diff(data(:,2:2:end),1,2))],2);

temp = [f1,f2,f3,f4,f5,f6,f7,f8];
for i = 1:num_gp
    features(i,:)=reshape(temp((i-1)*num_channel+1:i*num_channel,:),[1,8.*num_channel]);
end

end

