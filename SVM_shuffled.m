%% SVM, shuffled method
clear all
close all
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
load('ECGfeatures_100_47feat_16s');
load('score_anova')
load('score_wil')
load('seqmat')
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
for k = 1:1
for class_set = 1:6

[datamat,labelvec,subvec] = celldata2mat(feat_std,cate_feat);

% class_set = 6;
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

for trial = 1:1
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
% YTrain = removecats(YTrain);
% YTest = removecats(YTest);
% countcats(YTrain)
% countcats(YTest)

dimension_flag = 'non';
switch dimension_flag
    case 'pca'
        % PCA
%         k = 20; % let say we use only 15 features.
        [V,W,D] = pca(XTrain','NumComponents',10);
        XTrain = W';
        XTest = V'*(XTest - mean(XTest,2));
%         XTest = V'*(XTest' - mean(XTest',2));
%         dataset = W'; 
    case 'selection'
        %selection features
        N_anova = score_anova>=k;
        N_wil = score_wil>=k;
%         XTrain = XTrain(~abd_feat(:,class_set),:);
%         XTest = XTest(~abd_feat(:,class_set),:);
%         XTrain = XTrain(N_wil,:);
%         XTest = XTest(N_wil,:);
        XTrain = XTrain(history.In(k,:)',:);
        XTest = XTest(history.In(k,:)',:);
    case 'non'
end

% rng(); with optimal parameters
% t = templateSVM('Standardize',false,'KernelFunction','rbf',...
%     'KernelScale',0.013534,'BoxConstraint',939.78);
% SVMModel = fitcecoc(XTrain',YTrain','Learners',t);

t = templateSVM('Standardize',false,'KernelFunction','rbf',...
    'KernelScale','auto');
SVMModel = fitcecoc(XTrain',YTrain','Learners',t);

%optimization
% t = templateSVM('Standardize',false,'KernelFunction','polynomial'); 
% SVMModel = fitcecoc(XTrain',YTrain','Learners',t,...
%     'OptimizeHyperparameters','auto',...
%       'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%       'expected-improvement-plus'));

%one against one  
% SVMModel = fitcsvm(XTrain',YTrain','Standardize',false,'KernelFunction','polynomial',...
%     'OptimizeHyperparameters','auto',...
%       'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%       'expected-improvement-plus'));

CVSVMModel = crossval(SVMModel);
kfoldLoss(CVSVMModel)
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
[label,score] = predict(CompactSVMModel,XTest');
acc(trial) = sum(label == YTest')./numel(YTest);
% optimizedpara(:,class_set) = [SVMModel.BoxConstraints(1),CompactSVMModel.KernelParameters.Scale];
% plotconfusion(YTest',label)
C = confusionmat(YTest',label);
kappacoef(trial) = kappa(C);
end
% accavg(class_set) =  mean(acc)
% kappacoefave(class_set) = mean(kappacoef)
accavg(k,class_set) =  mean(acc)
kappacoefave(k,class_set) = mean(kappacoef)
end
end

%% plot?? shuffled 
accmat = zeros(6,6);
kappamat = zeros(6,6);

accmat(1,:)= [  0.8982    0.8752    0.8707    0.8936    0.9091    0.9229];
kappamat(1,:)=  [ 0.8470    0.8123    0.8056    0.7867    0.8168    0.8448];
accmat(2,:)= [ 0.8988    0.8806    0.8878    0.9119    0.9400    0.9339];
kappamat(2,:)=  [0.8476    0.8201    0.8312    0.8235    0.8798    0.8674];
accmat(3,:)= [ 0.9141    0.9048    0.8829    0.9229    0.9364    0.9202];
kappamat(3,:)=  [0.8703    0.8566    0.8239    0.8456    0.8725    0.8399];
accmat(4,:)= [  0.9055    0.8976    0.8945    0.9119    0.9355    0.9138];
kappamat(4,:)=  [0.8578    0.8455    0.8415    0.8231    0.8699    0.8268];
accmat(5,:)= [  0.8890    0.8994    0.8799    0.9156    0.9236    0.9119];
kappamat(5,:)=  [0.8327    0.8484    0.8189    0.8297    0.8468    0.8234];
accmat(6,:) = [0.9178    0.9103    0.8896    0.9229    0.9518    0.9404];
kappamat(6,:) = [0.8763    0.8647    0.8341    0.8450    0.9030    0.8805];
figure,
c = categorical({'PCA: 5','PCA: 10','PCA: 15', 'PCA: 20', 'Selection','No reduction'});
c = reordercats(c,{'PCA: 5','PCA: 10','PCA: 15', 'PCA: 20', 'Selection','No reduction'});
bar(c,kappamat(1:6,:));
ylim([0.7 1])
grid on
% hold on
% bar(c,kappamat(1:6,:));
title('Shuffle')
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');

%%
accno = accavg;
kappano = kappacoefave;
save('accno827.mat','accno','kappano');
%%
load('accno827');
kappamat = [kappano;kappacoefave];
accmat = [accno;accavg];

save('kappa_acc827_wil.mat','kappamat','accmat');
%%
% load('selection');

figure,
c = categorical({'No reduction','s = 1, n = 44','s = 2, n = 41','s= 3, n= 37','s = 4, n =32','s = 5, n = 26','s = 6, n = 17','s= 7, n = 11'});
c = reordercats(c,{'No reduction','s = 1, n = 44','s = 2, n = 41','s= 3, n= 37','s = 4, n =32','s = 5, n = 26','s = 6, n = 17','s= 7, n = 11'});
subplot(2,1,1)
bar(c,accmat(1:8,:));
                x0=10;
                y0=10;
                width=500;
                height=400;
                set(gcf,'units','points','position',[x0,y0,width,height]) 
ylim([0.6 1])
ylabel('Accuracy')
grid on
% hold on
% bar(c,kappamat(1:6,:));
title('Feature selection by Wilcoxon test')
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');

subplot(2,1,2)
bar(c,kappamat(1:8,:));
ylim([0.4 1])
ylabel('Kappa')
grid on
% hold on
% bar(c,kappamat(1:6,:));
title('Feature selection by Wilcoxon test')
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');


%%
load('sequanselacc')
xaxis = fliplr(2:47);
figure,
subplot(2,1,1)
plot(xaxis,accavg,'Marker','.'); 
    x0=10;
    y0=10;
    width=500;
    height=400;
    set(gcf,'units','points','position',[x0,y0,width,height])
vline(14,'--k','max: n = 14')
% plot(kappacoefave,'Marker','.','LineStyle','--');
% hold off
title("Classification accuracy in 'shuffled' scenario")
grid on 
xlabel('Number of features (n)')
ylabel('Accuracy')
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');

subplot(2,1,2)
plot(xaxis,kappacoefave,'Marker','.'); 
vline(14,'--k','max: n = 14')
title("Cohen's kappa in 'shuffled' scenario")
grid on 
xlabel('Number of features (n)')
ylabel("Cohen's kappa")
legend('B P S','B P M','B P S/M', 'B P', 'B S/M','P S/M','Location','northeastoutside');

%% 
for i = 1:7
    temp222(i) = sum(score_wil>=i);
end
%% bar chart, linear, polynomial, rbf
