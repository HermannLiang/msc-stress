clear
clc
close all
% examine the features, with visualisation enable
subjectno = 4;  %no. of subject
vis = 0;
para.T_winl = 250;
para.T_incre = 10;
para.F_winl = 250;
para.F_incre = 10;
para.F_pwinl = 150;
para.F_over = 10;
para.N_winl = 250;
para.N_incre = 10;
% input subject number, output features.
for subjectno = 1:4
[feat{1,subjectno},cate_feat{1,subjectno}] = getfeatures(subjectno,para,vis);
end
%standardizing: remove the median values from rest stage ??
%
%Selecting features: analysis of variance, p-test, anova, GA, 
%construct the machine learning network




%% Sequence to sequence classification example 

XTrain = feat(1:3);
YTrain = cate_feat(1:3);
XTest = feat(4);
YTest = cate_feat(4);

X = XTrain{1}(2,:);
classNames = categories(YTrain{1});

mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end

mu = mean([XTest{:}],2);
sig = std([XTest{:}],0,2);

for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sig;
end

figure
plot(XTrain{3}')
xlabel("Time Step")
title("Training Observation 1")
legend("Feature " + string(1:length(mu)),'Location','northeastoutside')

save('XTrain.mat','XTrain');save('YTrain.mat','YTrain');
save('XTest.mat','XTest');save('YTest.mat','YTest');
%%
clear all
load('XTrain.mat');load('YTrain.mat');load('XTest.mat');load('YTest.mat')
featureDimension = 26;
numHiddenUnits = 100;
numClasses = 5;

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
    'LearnRateDropPeriod',20, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

%%
figure,
plot(XTest{1})
xlabel("Time Step")
legend("Feature " + (1:featureDimension))
title("Test Data")

YPred = classify(net,XTest{1});
acc = sum(YPred == YTest{1})./numel(YTest{1})

figure
plot(YPred,'.-')
hold on
plot(YTest{1})
hold off

xlabel("Time Step")
ylabel("Activity")
title("Predicted Activities")
legend(["Predicted" "Test Data"])
%%
sequence
