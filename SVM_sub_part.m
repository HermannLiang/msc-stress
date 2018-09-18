%% SVM, partitioning data based on subject
clear all
close all
% addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
load('seqmat')
% load('ECGfeatures_50_20feat');
% load('ECGfeatures_100_20feat_16s');
load('ECGfeatures_100_47feat_16s');
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

% 3 3-class classification and 3 binary-classification
    % Partition: subjectbase
total = numel(feat); % total subject in the dataset
no_train = 15;
no_test = total - no_train;
div = 1:16;
% div = randperm(total); 
for k = 1:1
for class_set = 1:6
    [datamat,labelvec,subvec] = celldata2mat(feat_tanh,cate_feat);
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
subjec_m = subvec(:,~isundefined(idx_m));
countcats(cate_m)
categories(cate_m)

for trial = 1:16
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

dimension_flag = 'pca';
switch dimension_flag
    case 'pca'
        % PCA
%         k = 47; % let say we use only 15 features.
        [V,W,D] = pca(XTrain','NumComponents',20);
        XTrain = W';
        XTest = V'*(XTest - mean(XTest,2));
    case 'selection'
        %selection features
%         XTrain = XTrain(~abd_feat(:,class_set),:);
%         XTest = XTest(~abd_feat(:,class_set),:);
        XTrain = XTrain(history.In(34,:)',:);
        XTest = XTest(history.In(34,:)',:);
    case 'non'
end

     clearvars SVMModel CompactSVMModel CVSVMModel label
%SVM Training 
% rng(23);
% c = cvpartition(YTrain,'kFold',10);
% t = templateSVM('Standardize',false,'KernelFunction','rbf',... % no optimization
%     'KernelScale',0.022021,'BoxConstraint',631.4686);
t = templateSVM('Standardize',false,'KernelFunction','rbf',... % no optimization
    'KernelScale','auto');
SVMModel = fitcecoc(XTrain',YTrain','Learners',t);
% t = templateSVM('Standardize',false,'KernelFunction','RBF'); %optimization
% SVMModel = fitcecoc(XTrain',YTrain','Learners',t,...
%     'OptimizeHyperparameters','auto',...
%       'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%       'expected-improvement-plus'));
CVSVMModel = crossval(SVMModel);
kfoldLoss(CVSVMModel);
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
[label,score] = predict(CompactSVMModel,XTest');
acc(trial) =  sum(label == YTest')./numel(YTest);
% subplot(4,4,trial)
% figure,
% plotconfusion(YTest',label);
C = confusionmat(YTest',label);
kappacoef(trial) = kappa(C);
end
% avgacc(k,class_set) = mean (acc)
% kappacoefavg(k,class_set) = mean(kappacoef)
end 
end
% kappa(C);

% acc
%% Plot the data
accmat(1,:) = [0.5257    0.5261    0.4911    0.5739    0.6356    0.7224];
accmat(2,:) = [0.5574    0.5916    0.5801    0.6818    0.7520    0.7527];
accmat(3,:) = [0.5431    0.5017    0.5264    0.5741    0.6508    0.7153];
accmat(4,:) = [0.5357    0.5555    0.5203    0.6166    0.7231    0.7628];
accmat(5,:) = [0.5129    0.4942    0.4681    0.5892    0.6210    0.7171];
accmat(6,:) = [0.5074    0.5225    0.5076    0.5871    0.6695    0.7118];
accmat(7,:) = [0.5280    0.4934    0.5139    0.5727    0.6420    0.7275];
accmat(8,:) = [0.5062    0.5055    0.4870    0.5605    0.6835    0.7499];
accmat(9,:) = [0.5125    0.4911    0.4948    0.5805    0.6750    0.7171];
accmat(10,:) = [0.5183    0.4826    0.5033    0.5295    0.6665    0.7283];
barlabel = {'all, leave 1 out','pca, leave 1 out',...
    'all, leave 2 out','pca, leave 2 out',...
    'all, leave 3 out','pca, leave 3 out',...
    'all, leave 4 out','pca, leave 4 out',...
    'all, leave 5 out','pca, leave 5 out'};
c = categorical(barlabel);
c= reordercats(c,barlabel);
figure,
bar(c,accmat);
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');
ylabel('accuracy')
grid on
                x0=10;
                y0=10;
                width=500;
                height=300;
                set(gcf,'units','points','position',[x0,y0,width,height]) 
% 1, non, leave 1 out 
% 2, pca leave 1 out
% 3, non, leave 2 out
% 4, pca leave 2 out
% 5, non, leave 3 out
% 6, pca leave 3 out
% 7, non, leave 4 out
% 8, pca leave 4 out
% 9, non, leave 5 out
% 10, pca leave 5 out
%% PCA vs non ??
% save('Leaveoneoutacckappa.mat','avgacc','kappacoefavg');
load('Leaveoneoutacckappa')
figure,
subplot(2,1,1)
plot(avgacc,'Marker','.'); 
    x0=10;
    y0=10;
    width=500;
    height=400;
    set(gcf,'units','points','position',[x0,y0,width,height])
% vline(6,'--k')
% plot(kappacoefave,'Marker','.','LineStyle','--');
% hold off
title('Leave-one-out')
grid on 
xlabel('Dimension of subspace (k)')
ylabel('Accuracy')
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');

subplot(2,1,2)
plot(kappacoefavg,'Marker','.'); 
% vline(6,'--k')
title("Leave-one-out")
grid on 
xlabel('Dimension of subspace (k)')
ylabel("Cohen's kappa")
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');
%%
% save('16subleaveoneout.mat','acc','kappacoef')
load('16subleaveoneout')
figure,
% subplot(2,1,1)
plot(acc(:,4:6),'x','LineWidth',2); 
    x0=10;
    y0=10;
    width=500;
    height=400;
    set(gcf,'units','points','position',[x0,y0,width,height])
% vline(6,'--k')
% plot(kappacoefave,'Marker','.','LineStyle','--');
% hold off
% title('')
grid on 
xlabel('Subject index')
xticks(1:16)
ylabel('Stress level')
hold on
p = plot(1:16,ones(1,16)*median(acc(:,4)),'--');
p.Color = [0    0.4470    0.7410];
p = plot(1:16,ones(1,16)*median(acc(:,5)),'--r');
p.Color =[0.8500    0.3250    0.0980];
p = plot(1:16,ones(1,16)*median(acc(:,6)),'--y');
p.Color = [0.9290    0.6940    0.1250];
% legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');
legend('B P', 'B S/M','P S/M','median of B P','median of B S/M','median of P S/M','Location','northeastoutside');
hold off
% subplot(2,1,2)
% plot(kappacoef,'x'); 
% % vline(6,'--k')
% title("Leave-one-out")
%0    0.4470    0.7410
    %0.8500    0.3250    0.0980
    %0.9290    0.6940    0.1250
% grid on 
% xlabel('Dimension of subspace (k)')
% ylabel("Cohen's kappa")
% legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');
%% TOTAL method
% PCA 
 clearvars SVMModel CompactSVMModel CVSVMModel XTest YTest label
k = 10; % let say we use only 15 features.
[V,W,D] = pca(feat_m','NumComponents',k);
% now our new k-dimension dataset will be
pcaflag = 1;
if pcaflag ==1 
    dataset = W'; 
else
    dataset = feat_m;
    
end
%Total partition
datasize = size(dataset,2);
% partition_idx = randsample(datasize,round(0.8*datasize));
partition_idx = randperm(datasize);
train_idx = partition_idx(1:round(0.8*datasize));
test_idx = partition_idx(round(0.8*datasize)+1:end);
XTrain = dataset(:,train_idx);
YTrain = cate_m(:,train_idx);
XTest = dataset(:,test_idx);
YTest = cate_m(:,test_idx);
YTrain = removecats(YTrain);
YTest = removecats(YTest);

rng(23);
t = templateSVM('Standardize',false,'KernelFunction','rbf',...
    'KernelScale','auto');
SVMModel = fitcecoc(XTrain',YTrain','Learners',t);
CVSVMModel = crossval(SVMModel);
kfoldLoss(CVSVMModel)
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
[label,score] = predict(CompactSVMModel,XTest');
acc =  sum(label == YTest')./numel(YTest)
plotconfusion(YTest',label)
C = confusionmat(YTest',label);
kappacoef = kappa(C);

%%




% feat_tanh = feat_std;
total = numel(feat); % total subject in the dataset
div = randperm(total); 
no_train = 10;
no_test = total - no_train;
f_anova = 0; %feature selection by anova


for trial = 1:12
    clearvars SVMModel CompactSVMModel CVSVMModel XTest YTest label
div = circshift(div,trial);
% div = randperm(total); 
cat_B = [];
cat_A= [];
cat_R = [];
% only extracts rest, anticipate and stress

for i = 1:no_train
    B = setcats(cate_feat{div(i)},{'resting'});
    A = setcats(cate_feat{div(i)},{'alarm'});
    R = setcats(cate_feat{div(i)},{'resistance'});
    feat_B{i} = feat_tanh{div(i)}(:,~isundefined(B));
    cat_B = [cat_B,cate_feat{div(i)}(:,~isundefined(B))];
    feat_A{i} = feat_tanh{div(i)}(:,~isundefined(A));
    cat_A = [cat_A,cate_feat{div(i)}(:,~isundefined(A))];
    feat_R{i} = feat_tanh{div(i)}(:,~isundefined(R));
    cat_R = [cat_R,cate_feat{div(i)}(:,~isundefined(R))];
end
Bmat = cell2mat(feat_B);
Amat = cell2mat(feat_A);
Rmat = cell2mat(feat_R);
if f_anova == 1
Bmat(abd_feat,:) = [];
Rmat(abd_feat,:) = [];
Amat(abd_feat,:) = [];
end
XTrain = [Bmat,Amat,Rmat]';
YTrain = [cat_B,cat_A,cat_R]';
YTrain = removecats(YTrain);

clearvars B A R feat_B feat_A feat_R Rmat Bmat Amat

cat_B = [];
cat_A= [];
cat_R = [];
% only extracts rest, anticipate and stress
for i = 1:no_test
    Bt = setcats(cate_feat{div(i+no_train)},{'resting'});
    At = setcats(cate_feat{div(i+no_train)},{'alarm'});
    Rt = setcats(cate_feat{div(i+no_train)},{'resistance'});
    feat_B{i} = feat_tanh{div(i+no_train)}(:,~isundefined(Bt));
    cat_B = [cat_B,cate_feat{div(i+no_train)}(:,~isundefined(Bt))];
    feat_A{i} = feat_tanh{div(i+no_train)}(:,~isundefined(At));
    cat_A = [cat_A,cate_feat{div(i+no_train)}(:,~isundefined(At))];
    feat_R{i} = feat_tanh{div(i+no_train)}(:,~isundefined(Rt));
    cat_R = [cat_R,cate_feat{div(i+no_train)}(:,~isundefined(Rt))];
end

Bmat = cell2mat(feat_B);
Amat = cell2mat(feat_A);
Rmat = cell2mat(feat_R);
if f_anova == 1
Bmat(abd_feat,:) = [];
Rmat(abd_feat,:) = [];
Amat(abd_feat,:) = [];
end
XTest = [Bmat,Amat,Rmat]';
YTest = [cat_B,cat_A,cat_R]';
YTest = removecats(YTest);

%%
rng(23);
t = templateSVM('Standardize',false,'KernelFunction','RBF',...
    'KernelScale','auto');
SVMModel = fitcecoc(XTrain,YTrain,'Learners',t);
CVSVMModel = crossval(SVMModel);
kfoldLoss(CVSVMModel)
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
[label,score] = predict(CompactSVMModel,XTest);
table(YTest(1:10),label(1:10),score(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'})
acc4(trial) =  sum(label == YTest)./numel(YTest)
end

acc4avg = mean(acc4)
plotconfusion(YTest,label)

%%
