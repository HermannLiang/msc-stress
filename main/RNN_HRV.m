%%%% RNN-LSTM, training HRV directly
clear all
close all

addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
load('ECGfeatures_100_47feat_16s');

% pre processing data

for i = 1:numel(cate_hrv)
    B = setcats(cate_hrv{i},{'baseline','preparation','speech'});
    feat_3{i} = hrv{i}(:,~isundefined(B));
    cate_hrv_3{i} = B(:,~isundefined(B));
end

total = numel(hrv);
no_train = 15;
no_test = total - no_train;

div = 1:16;
for trial = 1:16
    div = circshift(div,trial);
XTrain = feat_3(div(1:no_train));
YTrain = cate_hrv_3(div(1:no_train));
XTest = feat_3(div(no_train+1:end));
YTest = cate_hrv_3(div(no_train+1:end));

X = XTrain{1}(1,:);
classNames = categories(YTrain{1});

%%%% z-score normalization%%%%%%%%%
mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);
% for i = 1:numel(XTrain)
%     XTrain{i} = 0.5.*(tanh(0.01.*(XTrain{i}-mu)./sig)+1);
% end
for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end
%tanh
mu = mean([XTest{:}],2);
sig = std([XTest{:}],0,2);
% for i = 1:numel(XTest)
%     XTest{i} = 0.5.*(tanh(0.01.*(XTest{i}-mu)./sig)+1);
% end
for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sig;
end
%%%%%%%%%%%%%%%%%%%%%%%
% figure
% plot(XTrain{1}')
% xlabel("Time Step")
% title("Training Observation 1")
% legend("Feature " + string(1:length(mu)),'Location','northeastoutside')

%
featureDimension = 1;
numHiddenUnits = 100;
numClasses = 3;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',2,...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',30,...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);


YPred(trial) = classify(net,XTest);
%     'MiniBatchSize',miniBatchSize, ...
%     'SequenceLength','longest');
figure,
plotconfusion(YTest{1},YPred{trial});
% plotconfusion(YTest,YPred{1});
title(['Subject ' num2str(trial)])
acc(trial) = sum(YPred{trial} == YTest{1})./numel(YTest{1})
end

[mean(acc), std(acc), max(acc), min(acc)]
%%  HRV RNN sequence to label classification
% [XTrain,YTrain] = japaneseVowelsTrainData;
% XTrain(1:5)

clear all
close all
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
load('ECGfeatures_100_47feat_16s');

%pre processing data
hrv_3 = [];
cate_hrv_3 = [];
hrv_idx = [];
idx = 1;
for i = 1:numel(cate_hrv)
    B1 = setcats(cate_hrv{i},{'baseline'});
    B2= setcats(cate_hrv{i},{'preparation'});
    B3 = setcats(cate_hrv{i},{'speech'});
    hrv_temp =  {hrv{i}(:,~isundefined(B1)),hrv{i}(:,~isundefined(B2)),hrv{i}(:,~isundefined(B3))};
    hrv_3 = [hrv_3, hrv_temp];
%     cate_temp = [B1(:,~isundefined(B1)),B2(:,~isundefined(B2)),B3(:,~isundefined(B3))];
    cate_temp = categorical({'baseline','preparation','speech'});
    cate_hrv_3 = [cate_hrv_3,cate_temp];
%     hrv_idx = [hrv_idx, idx*ones(1,length(B1(:,~isundefined(B1)))),...
%         (idx+1)*ones(1,length(B2(:,~isundefined(B2)))),...
%         (idx+2)*ones(1,length(B3(:,~isundefined(B3))))];
%     idx = idx+3;
end

% partition_idx = randperm(48);
partition_idx = 1:48;
train_idx = partition_idx(1:36);
test_idx = partition_idx(36+1:end);
XTrain = hrv_3(train_idx);
XTest = hrv_3(test_idx);
YTrain = cate_hrv_3(partition_idx <37);
YTest = cate_hrv_3(partition_idx >=37);
% YTrain = cate_hrv_3(hrv_idx ==train_idx);
% [~,idxsIntoA] = intersect(train_idx,hrv_idx,'stable');
% idx_x = arrayfun(@(x)find(hrv_idx==x,1),train_idx);
% idx1 = zeros(1,length(hrv_idx));
% idx2 = zeros(1,length(hrv_idx));
% for j = 1:length(hrv_idx)
%     [~,a] = intersect(50,hrv_idx(j),'stable');
%     if intersect(train_idx,hrv_idx(j),'stable') == 1
%         idx1(j) = 1;
%     end
%     if intersect(test_idx,hrv_idx(j),'stable') == 1
%         idx2(j) = 1;
%     end
% end


%%
figure
plot(XTrain{3}')
xlabel("Time Step")
title("Training Observation 1")
legend("Feature " + string(1:1),'Location','northeastoutside')

numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);
miniBatchSize = 2;

figure
bar(sequenceLengths)
ylim([0 max(sequenceLengths)])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

inputSize = 1;
numHiddenUnits = 100;
numClasses = 3;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 50;
% miniBatchSize = 32;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain',YTrain',layers,options);

%%
numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);

% miniBatchSize = 27;
YPred = classify(net,XTest', ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
plotconfusion(YTest',YPred);
acc = sum(YPred == YTest')./numel(YTest)

%%
[XTest1,YTest1] = japaneseVowelsTestData;