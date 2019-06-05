%% Sequence to sequence classification, 3 stages comparisons.
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
% pre-processing 
clear all
close all
% load('ECGfeatures');
load('ECGfeatures_100_47feat_16s');
% load('abd_feat')

mu = mean([feat{:}],2);
sig = std([feat{:}],0,2);

for i = 1:numel(feat)
    feat_std{i} = (feat{i} - mu) ./ sig;
end
% tanh normalization

for i = 1:numel(feat)
    feat_tanh{i} = 0.5.*(tanh(0.01.*(feat{i}-mu)./sig)+1);
end

% tanh was not good enough in LSTM
% only extracts rest, anticipate and stress
class_set = 6;
for i = 1:numel(cate_feat)
    switch class_set
        case 1
    B = setcats(cate_feat{i},{'baseline','preparation','speech'});
        case 2
    B = setcats(cate_feat{i},{'baseline','preparation','math'});
        case 3
     tempcat = mergecats(cate_feat{i},{'speech','math'},'S+M');
     B = setcats(tempcat,{'baseline','preparation','S+M'});
        case 4
    B = setcats(cate_feat{i},{'baseline','preparation'});
        case 5
    B = setcats(cate_feat{i},{'baseline','speech'});
        case 6
    B = setcats(cate_feat{i},{'preparation','speech'});
    end
    feat_3{i} = feat_tanh{i}(:,~isundefined(B));
    cate_feat_3{i} = B(:,~isundefined(B));
end

total = numel(feat);
no_train = 15;
no_test = total - no_train;
div = 1:16;
for trial = 1:16
div = circshift(div,trial);
% div = randperm(total); 
XTrain = feat_3(div(1:no_train));
YTrain = cate_feat_3(div(1:no_train));
XTest = feat_3(div(no_train+1:end));
YTest = cate_feat_3(div(no_train+1:end));

classNames = categories(YTrain{1});

featureDimension = size(XTrain{1},1);
numHiddenUnits= 100;
numClasses = numel(classNames);

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    %change it to column vector??
%     lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',50,...
    'MiniBatchSize',32,...
    'Verbose',0, ...
    'Shuffle','never')
%     'plots','training-progress');
trial
net = trainNetwork(XTrain,YTrain,layers,options);

YPred = classify(net,XTest{1});
acc(trial) = sum(YPred == YTest{1})./numel(YTest{1})

plot_flag = 1;
if plot_flag ==1
figure(1)
plotconfusion(YPred,YTest{1})

figure(2)
plot(YPred,'.-')
hold on
plot(YTest{1})
hold off
xlabel("Time Step")
ylabel("State")
title("Predicted states")
legend(["Predicted" "Test Data"])
end

end
[mean(acc), std(acc), max(acc), min(acc)]
%%


% Test another 
% figure,
% plot(XTest{2})
% xlabel("Time Step")
% legend("Feature " + (1:featureDimension))
% title("Test Data")

% YPred = classify(net,XTest{2});
% acc = sum(YPred == YTest{2})./numel(YTest{2})
% 
% figure
% plot(YPred,'.-')
% hold on
% plot(YTest{2})
% hold off
% 
% xlabel("Time Step")
% ylabel("State")
% title("Predicted States")
% legend(["Predicted" "Test Data"])

%%%%%%%%%%%end %%%%%%%%%%%%%%%%%%%%%%%%
%% NEW RNN training
clear all
close all
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
load('ECGfeatures_100_47feat_16s');
load('seqmat')
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
[datamat,labelvec,subvec] = celldata2mat(feat_std,cate_feat);
total = numel(feat); % total subject in the dataset
no_train = 15;
no_test = total - no_train;
div = 1:16;
trial = 1;
class_set = 4;
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

div = circshift(div,trial);

% PCA dimension reduction + leave one out
% div = randperm(total); 
train_idx = [];
test_idx = [];
for i = 1:no_train
    train_idx = [train_idx, find(subjec_m == div(i))];
end
for i = 1:no_test
    test_idx = [test_idx, find(subjec_m == div(i+no_train))];
end
XTrain = feat_m(:,train_idx);
YTrain = cate_m(:,train_idx);
XTest = feat_m(:,test_idx);
YTest = cate_m(:,test_idx);
countcats(YTrain)
countcats(YTest)

featureDimension = size(XTrain,1);
numHiddenUnits= 100;
numClasses = numel(categories(cate_m));

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%     lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',50,...
    'MiniBatchSize',5,...
    'Verbose',0, ...
    'Shuffle','never',...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

YPred = classify(net,XTest);
acc(trial) = sum(YPred == YTest)./numel(YTest)

plot_flag = 1;
if plot_flag ==1
figure(1)
plotconfusion(YPred,YTest)

figure(2)
plot(YPred,'.-')
hold on
plot(YTest)
hold off
xlabel("Time Step")
ylabel("State")
title("Predicted states")
legend(["Predicted" "Test Data"])
end

mean(acc)

% numNN = 10;
% NN = cell(1, numNN);
% perfs = zeros(1, numNN);
% for i = 1:numNN
%   fprintf('Training %d/%d\n', i, numNN);
%   NN{i} = trainNetwork(XTrain,YTrain,layers,options);
%   y2 = NN{i}(XTest);
%   perfs(i) = mse(net, XTest, y2);
% end

%%

YPred = classify(net,XTest);
acc = sum(YPred == YTest)./numel(YTest)
plotconfusion(YPred,YTest)

