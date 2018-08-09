%%
sequence

%% LOAD HRV DATA
% first argument, visualization, 1 or 0.
% second argument
%% ADD FOLDERS
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
%%
% filtermethod, 0: no filtering; 
%                           1: complex 3-stage filtering; default
%                           2: median filtering 
% different recording with different methods:

close all
clear all
[hrv{1},stage{1}] = gethrv(1,1,1);
[hrv{2},stage{2}] = gethrv(2,1,1);
[hrv{3},stage{3}] = gethrv(3,1,1);
[hrv{4},stage{4}] = gethrv(4,1,1);
[hrv{5},stage{5}] = gethrv(5,1,1);
[hrv{6},stage{6}] = gethrv(6,1,1);
[hrv{7},stage{7}] = gethrv(7,1,0);
[hrv{8},stage{8}] = gethrv(8,1,1);
[hrv{9},stage{9}] = gethrv(9,1,1);
[hrv{9},stage{9}] = gethrv(9,1,1);
[hrv{10},stage{10}] = gethrv(10,1,1);
% save('hrv.mat')
%% EXTRACT FEATURES
clear all
close all
load('hrv.mat')
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
for subjectno = 1:10
[feat{1,subjectno},feat_names,cate_feat{1,subjectno},cate_hrv{1,subjectno}] = getfeatures(subjectno,hrv{subjectno},stage{subjectno},para,vis);
disp('Extracted features from subject ' + string(subjectno))
end
save('ECGfeatures');
%standardizing: remove the median values from rest stage ??
%
%Selecting features: analysis of variance, p-test, anova, GA, 
%construct the machine learning network


%% ANOVA, USING ORIGINAL DATA
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

feat_names{abd_feat}

%% Normalization of data, two techniques

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

%% SVM CLASSIFICATION, RBF, Preparing Data

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

% remove excessive features?? 
Bmat(abd_feat,:) = [];
Rmat(abd_feat,:) = [];
Amat(abd_feat,:) = [];
%% BASELINE VS RESISTANCE
clearvars SVMModel CompactSVMModel CVSVMModel X Y XTest YTest label
rng(1); % For reproducibility
% Baseline vs Resistance classification
X = [Bmat,Rmat]';
Y = [cat_B,cat_R]';
Y = removecats(Y);
SVMModel = fitcsvm(X,Y,'Standardize',false,'KernelFunction','RBF',...
    'KernelScale','auto');
%X: observation*dimesion matrix, Y: observation* label
CVSVMModel = crossval(SVMModel,'Holdout',0.15);
kfoldLoss(CVSVMModel)
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = X(testInds,:);
YTest = Y(testInds,:);
[label,score] = predict(CompactSVMModel,XTest);
table(YTest(1:10),label(1:10),score(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'})
acc1 =  sum(label == YTest)./numel(YTest)
plotconfusion(YTest,label)
%% BASELINE VS ALARM

clearvars SVMModel CompactSVMModel CVSVMModel X Y XTest YTest label
rng(1); % For reproducibility
% Baseline vs Resistance classification
X = [Bmat,Amat]';
Y = [cat_B,cat_A]';
Y = removecats(Y);
SVMModel = fitcsvm(X,Y,'Standardize',false,'KernelFunction','RBF',...
    'KernelScale','auto');
%X: observation*dimesion matrix, Y: observation* label
CVSVMModel = crossval(SVMModel,'Holdout',0.15);
kfoldLoss(CVSVMModel)
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = X(testInds,:);
YTest = Y(testInds,:);
[label,score] = predict(CompactSVMModel,XTest);
table(YTest(1:10),label(1:10),score(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'})
acc2 =  sum(label == YTest)./numel(YTest)
plotconfusion(YTest,label)
%% ALARM VS RESISTANCE
clearvars SVMModel CompactSVMModel CVSVMModel X Y XTest YTest label
rng(1); % For reproducibility
% Baseline vs Resistance classification
X = [Amat,Rmat]';
Y = [cat_A,cat_R]';
Y = removecats(Y);
SVMModel = fitcsvm(X,Y,'Standardize',false,'KernelFunction','RBF',...
    'KernelScale','auto');
%X: observation*dimesion matrix, Y: observation* label
CVSVMModel = crossval(SVMModel,'Holdout',0.15);
kfoldLoss(CVSVMModel)
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = X(testInds,:);
YTest = Y(testInds,:);
[label,score] = predict(CompactSVMModel,XTest);
table(YTest(1:10),label(1:10),score(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'})
acc3 =  sum(label == YTest)./numel(YTest)

plotconfusion(YTest,label)

%% SVM+ECOC
clearvars SVMModel CompactSVMModel CVSVMModel X Y XTest YTest label
X = [Bmat,Amat,Rmat]';
Y = [cat_B,cat_A,cat_R]';
Y = removecats(Y);
rng(23);
t = templateSVM('Standardize',false,'KernelFunction','RBF',...
    'KernelScale','auto');
SVMModel = fitcecoc(X,Y,'Learners',t);
CVSVMModel = crossval(SVMModel,'Holdout',0.15);
kfoldLoss(CVSVMModel)
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = X(testInds,:);
YTest = Y(testInds,:);
[label,score] = predict(CompactSVMModel,XTest);
table(YTest(1:10),label(1:10),score(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'})
acc4 =  sum(label == YTest)./numel(YTest)
plotconfusion(YTest,label)

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

