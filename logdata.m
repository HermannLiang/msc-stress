clear
clc
close all
% examine the features, with visualisation enable
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
for subjectno = 6:6
[feat{1,subjectno},feat_names,cate_feat{1,subjectno},hrv{1,subjectno},cate_hrv{1,subjectno}] = getfeatures(subjectno,para,vis);
disp('Extracted features from subject ' + string(subjectno))
end
% save('ECGfeatures');
%standardizing: remove the median values from rest stage ??
%
%Selecting features: analysis of variance, p-test, anova, GA, 
%construct the machine learning network
%%
sequence

%%
% filtermethod, 0: no filtering; 
%                           1: complex 3-stage filtering; default
%                           2: median filtering 
hrv = gethrv(5,1,1);

%% Sequence to sequence classification example 
clear all
load('ECGfeatures')
div = randperm(6); 
XTrain = feat(div(1:4));
YTrain = cate_feat(div(1:4));
XTest = feat(div(5:6));
YTest = cate_feat(div(5:6));

X = XTrain{1}(2,:);
classNames = categories(YTrain{1});

%%%% z-score normalization%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%
feat_names = {'meanHR','sdHR','meanRR','sdRR','RMSSD','NN50','pNN50',...
    'iA_LF','iA_HF','pLF','pHF','LFtoHF',...
    'SampEn1','SampEn2','SampEn3','SampEn4',...
    'MSE1','MSE2','MSE3','MSE4','MSE5',...
    'MFE1','MFE2','MFE3','MFE4','MFE5'};
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
legend("Feature " + string(1:length(mu)),'Location','northeastoutside')

feat_idx = 10;
X = XTrain{1}(feat_idx,:);
classNames = categories(YTrain{1});
figure
for j = 1:numel(classNames)
    label = classNames(j);
    idx = find(YTrain{1} == label);
    hold on
    plot(idx,X(idx))
end
hold off

xlabel("Time Step")
ylabel(string(feat_names{feat_idx}))
title("Training Sequence 1; " + string(feat_names{feat_idx}))
legend(classNames,'Location','northwest')


%%

featureDimension = 26;
numHiddenUnits = 100;
numClasses = 4;

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
% figure,
% plot(XTest{1})
% xlabel("Time Step")
% legend("Feature " + (1:featureDimension))
% title("Test Data")

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

% Test another 
% figure,
% plot(XTest{2})
% xlabel("Time Step")
% legend("Feature " + (1:featureDimension))
% title("Test Data")

YPred = classify(net,XTest{2});
acc = sum(YPred == YTest{2})./numel(YTest{2})

figure
plot(YPred,'.-')
hold on
plot(YTest{2})
hold off

xlabel("Time Step")
ylabel("Activity")
title("Predicted Activities")
legend(["Predicted" "Test Data"])

%%%%%%%%%%%end %%%%%%%%%%%%%%%%%%%%%%%%
%% SVM part, starting from the example

% multiclass, for example ,stage 2,3,4
clear all
load fisheriris
X = meas(:,3:4);
Y = species;

%%
clear all
load('XTrain.mat');load('YTrain.mat');

X_ = [XTrain{1}';XTrain{2}';XTrain{3}';XTrain{4}']; %use two subjects for example
Y_ = [YTrain{1}';YTrain{2}';YTrain{3}';YTrain{4}']; 
[~,~,cate_idx] = unique(Y_);
X = X_((cate_idx == 2)|(cate_idx == 3)|(cate_idx == 4),:);
Y = Y_((cate_idx == 2)|(cate_idx == 3)|(cate_idx == 4),:);
% Y= mat2cell(Ycat,[1 1]);
%%
figure
gscatter(X(:,1),X(:,2),Y);
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Scatter Diagram of Iris Measurements}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend('Location','Northwest');

%%
SVMModels = cell(3,1);
classes = unique(Y);
rng(1); % For reproducibility

%%
for j = 1:numel(classes)
%     indx = strcmp(Y,classes(j)); % Create binary classes for each classifier
    indx = Y==classes(j);% Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
%%
% something wrong here
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

%%
figure
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
hold on
h(4:6) = gscatter(X(:,1),X(:,2),Y);
title('{\bf Iris Classification Regions}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend(h,{'setosa region','versicolor region','virginica region',...
    'observed setosa','observed versicolor','observed virginica'},...
    'Location','Northwest');
axis tight
hold off


%%

SVMModel = fitcsvm(X,y)

classOrder = SVMModel.ClassNames
%%
sv = SVMModel.SupportVectors;
figure
gscatter(X(:,2),X(:,3),y)
hold on
plot(sv(:,2),sv(:,3),'ko','MarkerSize',10)
legend('resting','resistance','Support Vector')
hold off
