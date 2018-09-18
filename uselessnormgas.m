%%mapping gas; your last effort
clear all
close all
load('ECGfeatures_100_47feat_16s');

% z-score normalization
mu = mean([feat{:}],2);
sig = std([feat{:}],0,2);
for i = 1:numel(feat)
    feat_std{i} = (feat{i} - mu) ./ sig;
end
% tanh normalization
for i = 1:numel(feat)
    feat_tanh{i} = 0.5.*(tanh(0.01.*(feat{i}-mu)./sig)+1);
end
subidx = 16;
A = 1:47;
featmins = [2,3,4,5,6,7,9,11,13,14,15:47];
Afilt=A(~ismember(A,featmins));
    featm = feat_tanh{subidx}(featmins,:);
    featp= feat_tanh{subidx}(Afilt,:);
    featm = ones(size(featm)) - featm;
    featnew = [featp;featm];

for i = 1:size(feat_tanh{subidx},2)
%     featnorm(i) = norm(feat_tanh{subidx}(:,i));
    featnorm(i) = norm(featnew(:,i));
end
plot(featnorm);
% hold on 
% feat1 = ones(1,size(feat_tanh{subidx},2))*2*3.4 - featnorm;
% plot(feat1)