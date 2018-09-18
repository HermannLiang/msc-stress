% ANOVA toolbox
% pre-processing 
clear all
load('ECGfeatures')
L = [cate_feat{1},cate_feat{2},cate_feat{3},cate_feat{4},cate_feat{5},cate_feat{6},cate_feat{7}];
%%
% z-score standarization
mu = mean([feat{:}],2);
sig = std([feat{:}],0,2);

for i = 1:numel(feat)
    feat_std{i} = (feat{i} - mu) ./ sig;
end
% tanh normalization

for i = 1:numel(feat)
    feat_tanh{i} = 0.5.*(tanh(0.01.*(feat{i}-mu)./sig)+1);
end

featidx = 7;
figure,
plot(feat_std{6}(featidx,:));
hold on
plot(feat_tanh{7}(featidx,:));
xlabel('Time step')
ylabel('normalized features')
legend('Z-score','tanh')
title('Features: ' + string(feat_names{featidx}))
save('feat_std.mat','feat_std');
save('feat_tanh.mat','feat_tanh');
%% CORRELATION TEST
featmat = cell2mat(feat_tanh);
[rho_corr,pval_corr] = corr(feat{1}');
% 1 stands for high correlation, lower than 0.05, 
% Because the p-value is less than the significance level of 0.05, 
% it indicates rejection of the hypothesis that no correlation exists between the two columns.
reject = pval_corr>0.05;
% [h,p] = ttest(featmat(1,:),featmat(2,:));
%histfit(featmat(1,:)) this is not normal distribution
%%
figure,
plot(featmat(6,:)); hold on
plot(featmat(11,:)); hold off

histfit(reshape(featmat,1,[]));

%% ANOVA
% pre-processing 
clear all
close all
load('ECGfeatures');
cat_B = [];
cat_A= [];
cat_R = [];
% only extracts rest, anticipate and stress
for i = 1:numel(cate_feat)
    B = setcats(cate_feat{i},{'resting'});
    A = setcats(cate_feat{i},{'alarm'});
    R = setcats(cate_feat{i},{'resistance'});
    feat_B{i} = feat{i}(:,~isundefined(B));
    cat_B = [cat_B,cate_feat{i}(:,~isundefined(B))];
    feat_A{i} = feat{i}(:,~isundefined(A));
    cat_A = [cat_A,cate_feat{i}(:,~isundefined(A))];
    feat_R{i} = feat{i}(:,~isundefined(R));
    cat_R = [cat_R,cate_feat{i}(:,~isundefined(R))];
end
Bmat = cell2mat(feat_B);
Amat = cell2mat(feat_A);
Rmat = cell2mat(feat_R);

for i = 1: numel(feat_names)
p_anova(i,1) = anova1([Bmat(i,:),Amat(i,:)],[ones(1,size(Bmat,2)),2*ones(1,size(Amat,2))],'off');
[~,p_ttest(i,1)] = ttest([Bmat(i,:),Amat(i,:)]);
end
for i = 1: numel(feat_names)
p_anova(i,2) = anova1([Bmat(i,:),Rmat(i,:)],[ones(1,size(Bmat,2)),2*ones(1,size(Rmat,2))],'off');
[~,p_ttest(i,2)] = ttest([Bmat(i,:),Rmat(i,:)]);
end
for i = 1: numel(feat_names)
p_anova(i,3) = anova1([Amat(i,:),Rmat(i,:)],[ones(1,size(Amat,2)),2*ones(1,size(Rmat,2))],'off');
[~,p_ttest(i,3)] = ttest([Amat(i,:),Rmat(i,:)]);
end

valid = p_anova<0.05/numel(feat_names); % reject null hypothesis at the bonferroni corrected p value
counts = sum(double(valid),2); % count n.o features that alt hypo holds
abd_feat = counts==0; % indices of features should be removed
save('abd_feat.mat','abd_feat');


%%

clear all
close all
load('ECGfeatures');
load('feat_tanh')
load('feat_std')
load('abd_feat')
cat_B = [];
cat_A= [];
cat_R = [];
% only extracts rest, anticipate and stress
for i = 1:numel(cate_feat)
    B = setcats(cate_feat{i},{'resting'});
    A = setcats(cate_feat{i},{'alarm'});
    R = setcats(cate_feat{i},{'resistance'});
    feat_B{i} = feat_tanh{i}(:,~isundefined(B));
    cat_B = [cat_B,cate_feat{i}(:,~isundefined(B))];
    feat_A{i} = feat_tanh{i}(:,~isundefined(A));
    cat_A = [cat_A,cate_feat{i}(:,~isundefined(A))];
    feat_R{i} = feat_tanh{i}(:,~isundefined(R));
    cat_R = [cat_R,cate_feat{i}(:,~isundefined(R))];
end
Bmat = cell2mat(feat_B);
Amat = cell2mat(feat_A);
Rmat = cell2mat(feat_R);

Bmat(abd_feat,:) = [];
Rmat(abd_feat,:) = [];
Amat(abd_feat,:) = [];
clearvars CVSVMModel
rng(1); % For reproducibility
X = [Bmat(i,:),Rmat(i,:)]';
Y = [cat_B,cat_R]';
CVSVMModel = fitcsvm(X,Y,'Holdout',0.15,'Standardize',false,'KernelFunction','RBF',...
    'KernelScale','auto');
%X: observation*dimesion matrix, Y: observation* label
% CVSVMModel = crossval(SVMModel);
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = X(testInds,:);
YTest = Y(testInds,:);

[label,score] = predict(CompactSVMModel,XTest);
table(YTest(1:10),label(1:10),score(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'})
acc =  sum(label == YTest)./numel(YTest)
