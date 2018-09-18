clear all
close all
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
load('ECGfeatures_100_47feat_16s');
load('score_anova')
load('score_wil')
%%
barflag = 0;  %plot anova barchart
% anova;
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
% [datamat,labelvec,subvec] = celldata2mat(feat_tanh,cate_feat);
% 3 3-class classification and 3 binary-classification
[datamat,labelvec,subvec] = celldata2mat(feat_tanh,cate_feat);

class_set = 3;
switch class_set
    case 1
        idx_m = setcats(labelvec,{'baseline','preparation','speech'});
    case 2
        idx_m = setcats(labelvec,{'baseline','preparation','math'});
    case 3
        labelvec = mergecats(labelvec,{'speech','math'},'S/M');
        idx_2 = setcats(labelvec,{'S/M'});
        idx_temp = 1:length(labelvec);
        idx_3 = idx_temp(~isundefined(idx_2));
        idx_4 = randsample(idx_3,floor(length(idx_3)/2));
        for j  = 1:length(idx_4)
            labelvec(idx_4(j)) = idx_2(1);
        end
        idx_m = setcats(labelvec,{'baseline','preparation','S/M'});
    case 4
        idx_m = setcats(labelvec,{'baseline','preparation'});
    case 5
        labelvec = mergecats(labelvec,{'speech','math'},'S/M');
        idx_2 = setcats(labelvec,{'S/M'});
        idx_temp = 1:length(labelvec);
        idx_3 = idx_temp(~isundefined(idx_2));
        idx_4 = randsample(idx_3,floor(length(idx_3)/2));
        for j  = 1:length(idx_4)
            labelvec(idx_4(j)) = idx_2(1);
        end
        idx_m = setcats(labelvec,{'baseline','S/M'});
    case 6
        labelvec = mergecats(labelvec,{'speech','math'},'S/M');
        idx_2 = setcats(labelvec,{'S/M'});
        idx_temp = 1:length(labelvec);
        idx_3 = idx_temp(~isundefined(idx_2));
        idx_4 = randsample(idx_3,floor(length(idx_3)/2));
        for j  = 1:length(idx_4)
            labelvec(idx_4(j)) = idx_2(1);
        end
        idx_m = setcats(labelvec,{'preparation','S/M'});
end
feat_m= datamat(:,~isundefined(idx_m));
cate_m = labelvec(:,~isundefined(idx_m));
cate_m = removecats(cate_m);
countcats(cate_m)
categories(cate_m)
subjec_m = subvec(:,~isundefined(idx_m));

total = numel(feat); % total subject in the dataset
no_train = 12;
no_test = total - no_train;
%  label :{'baseline','preparation','speech','math','recovery'});

    clearvars SVMModel CompactSVMModel CVSVMModel XTest YTest label
% div = randperm(total); 

%'Shuffled' partition
datasize = size(feat_m,2);
partition_idx = randperm(datasize);
train_idx = partition_idx(1:round(0.8*datasize));
test_idx = partition_idx(round(0.8*datasize)+1:end);
XTrain = feat_m(:,train_idx);
YTrain = cate_m(:,train_idx);
XTest = feat_m(:,test_idx);
YTest = cate_m(:,test_idx);
%%
XTrain = XTrain';
YTrain = YTrain';
XTest = XTest';
YTest = YTest';
%%
c = cvpartition(YTrain,'k',10);
t = templateSVM('Standardize',false,'KernelFunction','rbf',...
    'KernelScale','auto');
% SVMModel = fitcecoc(XTrain',YTrain','Learners',t);
opts = statset('display','iter');
classf = @(XTrain, YTrain, XTest, YTest)...
    sum(predict(fitcecoc(XTrain, YTrain,'Learners',t), XTest) ~= YTest);
[fs, history] = sequentialfs(classf, XTrain, YTrain, 'cv', c,'options', opts,'nfeatures',2,'direction', 'backward');
%%
% log
% Start forward sequential feature selection:
% Initial columns included:  none
% Columns that can not be included:  none
% Step 1, added column 3, criterion value 0.501235
% Step 2, added column 5, criterion value 0.246914
% Final columns included:  3 5 
% Start backward sequential feature selection:
% Initial columns included:  all
% Columns that must be included:  none
% Step 1, used initial columns, criterion value 0.191358
% Step 2, removed column 34, criterion value 0.181481
% Step 3, removed column 27, criterion value 0.17037
% Step 4, removed column 36, criterion value 0.161728
% Step 5, removed column 24, criterion value 0.155556
% Step 6, removed column 44, criterion value 0.149383
% Step 7, removed column 33, criterion value 0.148148
% Step 8, removed column 38, criterion value 0.141975
% Step 9, removed column 18, criterion value 0.139506
% Step 10, removed column 4, criterion value 0.141975
% Step 11, removed column 43, criterion value 0.140741
% Step 12, removed column 16, criterion value 0.139506
% Step 13, removed column 25, criterion value 0.138272
% Step 14, removed column 30, criterion value 0.135802
% Step 15, removed column 37, criterion value 0.135802
% Step 16, removed column 20, criterion value 0.134568
% Step 17, removed column 11, criterion value 0.134568
% Step 18, removed column 46, criterion value 0.130864
% Step 19, removed column 39, criterion value 0.12716
% Step 20, removed column 41, criterion value 0.12716
% Step 21, removed column 9, criterion value 0.123457
% Step 22, removed column 21, criterion value 0.125926
% Step 23, removed column 42, criterion value 0.123457
% Step 24, removed column 45, criterion value 0.118519
% Step 25, removed column 40, criterion value 0.120988
% Step 26, removed column 5, criterion value 0.120988
% Step 27, removed column 17, criterion value 0.118519
% Step 28, removed column 7, criterion value 0.123457
% Step 29, removed column 26, criterion value 0.122222
% Step 30, removed column 23, criterion value 0.122222
% Step 31, removed column 47, criterion value 0.117284
% Step 32, removed column 28, criterion value 0.118519
% Step 33, removed column 2, criterion value 0.118519
% Step 34, removed column 19, criterion value 0.117284
% Step 35, removed column 35, criterion value 0.117284
% Step 36, removed column 15, criterion value 0.118519
% Step 37, removed column 32, criterion value 0.124691
% Step 38, removed column 31, criterion value 0.125926
% Step 39, removed column 22, criterion value 0.119753
% Step 40, removed column 10, criterion value 0.118519
% Step 41, removed column 12, criterion value 0.108642
% Step 42, removed column 1, criterion value 0.120988
% Step 43, removed column 6, criterion value 0.12716
% Step 44, removed column 8, criterion value 0.145679
% Step 45, removed column 29, criterion value 0.155556
% Step 46, removed column 14, criterion value 0.250617
% Final columns included:  3 13 