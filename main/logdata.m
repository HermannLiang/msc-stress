%%
sequence
%% testing mcse
load('hrv.mat')
% [hrv{5},stage{5}] = gethrv(2,0,1);
hrv_ = hrv{5};
hrv_(400:end) = [];
% [ sd1, sd2]= poincare(hrv_);
% se = mmse(hrv_,2,1,0.15,10);
% samp = sampenc(hrv_,2,0.2);
tic
CosEn_ = cl_MCSE(hrv_,2,0.07,1,5); % 400 samples: 0.83,0.83*1688/60 = 23min
% 0.085334/200*5*100*4*800
toc
%% ADD FOLDERS
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
%% LOAD HRV DATA
% first argument, visualization, 1 or 0.
% second argument
% [hrv,stage] = gethrv(subjectno,display,filtermethod);
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
[hrv{7},stage{7}] = gethrv(7,1,1);
[hrv{8},stage{8}] = gethrv(8,1,1);
[hrv{9},stage{9}] = gethrv(9,1,1);
[hrv{10},stage{10}] = gethrv(10,1,1);
[hrv{11},stage{11}] = gethrv(11,1,1);
[hrv{12},stage{12}] = gethrv(12,1,1);
[hrv{13},stage{13}] = gethrv(13,1,1);
[hrv{14},stage{14}] = gethrv(14,1,1);
[hrv{15},stage{15}] = gethrv(15,1,1);
[hrv{16},stage{16}] = gethrv(16,1,1);
save('hrv.mat')
%% EXTRACT FEATURES
clear all
close all
load('hrv.mat')
vis = 1;
% para.T_winl = 100;
% para.T_incre = 10;
% para.F_winl = 100;
% para.F_incre = 10;
% % para.F_pwinl = 100;
% % para.F_over = 10;
% para.N_winl = 100;
% para.N_incre = 10;

para.T_winl = 100;
para.T_incre = 10;
para.F_winl = 100;
para.F_incre = 10;
para.N_winl = 100;
para.N_incre = 10;
tic
for subjectno = 4:4
[feat{1,subjectno},feat_names,cate_feat{1,subjectno},cate_hrv{1,subjectno}] = getfeatures(subjectno,hrv{subjectno},stage{subjectno},para,vis);
disp('Extracted features from subject ' + string(subjectno))
end
% save('ECGfeatures_100_38feat');
toc
% save('ECGfeatures_100_47feat_16s');
%standardizing: remove the median values from rest stage ??
%
%Selecting features: analysis of variance, p-test, anova, GA, 
%construct the machine learning network
%% Generate entropy plot with scale
clear all
load('ECGfeatures_100_37feat_16s');
% ECGfeatures_100_20feat_14sub
% figure,
mu = mean([feat{:}],2);
sig = std([feat{:}],0,2);
%z-score
for i = 1:numel(feat)
    feat_std{i} = (feat{i} - mu) ./ sig;
end
% tanh normalization
for i = 1:numel(feat)
    feat_tanh{i} = 0.5.*(tanh(0.01.*(feat{i}-mu)./sig)+1);
end
[datamat,labelvec,subvec] = celldata2mat(feat,cate_feat);

idx=isnan(datamat)| isinf(datamat);
[row,col] = find(idx);
%%
classnames = categories(cate_feat{1});
for i = 1:5
    idx_m = setcats(labelvec,classnames{i});
    feat_m= datamat(:,~isundefined(idx_m));
%     MSEmat = feat_m(19:28,:);
    MSEmat = feat_m(19:38,:);
%     MSEmat(isinf(MSEmat)) = [];
    for j = 1:20
        temp = MSEmat(j,:);
        temp(isinf(temp)) = [];
    MSEplot(j,i) = mean(temp,2);
    MSEsd(j,i) = std(temp);
    end
end
figure,
subplot(2,1,1)
errorbar(MSEplot,MSEsd,'-s','MarkerSize',2,...
    'MarkerEdgeColor','black','MarkerFaceColor','white');
% vline(10,'--k');
title('Multiscale sample entropy')
xlabel('scale')
ylabel('MSE')
legend('baseline','preparation','speech','math','recovery','Location','southeast');

%MFE
for i = 1:5
    idx_m = setcats(labelvec,classnames{i});
    feat_m= datamat(:,~isundefined(idx_m));
    MFEmat = feat_m(38:58,:);
%     MSEmat(isinf(MSEmat)) = [];
    for j = 1:20
        temp = MFEmat(j,:);
        temp(isinf(temp)) = [];
    MFEplot(j,i) = mean(temp,2);
    MFEsd(j,i) = std(temp);
    end
end
subplot(2,1,2)
errorbar(MFEplot,MFEsd,'-s','MarkerSize',2,...
    'MarkerEdgeColor','black','MarkerFaceColor','white');
% vline(8,'--k');
title('Multiscale fuzzy entropy')
xlabel('scale')
ylabel('MFE')
legend('baseline','preparation','speech','math','recovery','Location','southeast');

legend('baseline','preperation','speech','math','recovery','Location','southeast');
%% MCES plot 
classnames = categories(cate_feat{1});
for i = 1:5
    idx_m = setcats(labelvec,classnames{i});
    feat_m= datamat(:,~isundefined(idx_m));
    MCSEmat = feat_m(38:47,:);
%     MSEmat(isinf(MSEmat)) = [];
    for j = 1:10
        temp = MCSEmat(j,:);
        temp(isinf(temp)) = [];
    MCSEplot(j,i) = mean(temp,2);
    MCSEsd(j,i) = std(temp);
    end
end
figure,
errorbar(MCSEplot,MCSEsd,'-s','MarkerSize',2,...
    'MarkerEdgeColor','black','MarkerFaceColor','white');
% vline(10,'--k');
title('Multiscale Cosine Similarity entropy')
xlabel('scale')
ylabel('MCSE')
legend('baseline','preparation','speech','math','recovery','Location','southeast');
%% box plot, inspecting the data distribution for each subject
load('ECGfeatures_50_20feat');
subjectno = 12;
featureno = 20;
for subjectno = 1:13
% figure,
boxplot(feat{subjectno}(featureno,:),cate_feat{subjectno});
hold on
end
% title(['subject: ' num2str(subjectno) 'feature: ' feat_names{featureno}])

%% box plot 2, avearge all subject
% clear all
load('ECGfeatures_100_47feat_16s');
% ECGfeatures_100_20feat_14sub
% figure,
mu = mean([feat{:}],2);
sig = std([feat{:}],0,2);
%z-score
for i = 1:numel(feat)
    feat_std{i} = (feat{i} - mu) ./ sig;
end
% tanh normalization
for i = 1:numel(feat)
    feat_tanh{i} = 0.5.*(tanh(0.01.*(feat{i}-mu)./sig)+1);
end
[datamat,labelvec,subvec] = celldata2mat(feat,cate_feat);
i = 1;
figure,
    x0=500;
    y0=10;
    width=650;
    height=1250;
    set(gcf,'units','points','position',[x0,y0,width,height])
for featureno = [13:16,19,20,28,29,41,42]
subplot(5,2,i)
boxplot(datamat(featureno,:),labelvec);
% ylabel('Normalized features')
title (feat_names{featureno})
i = i+1;
end

% [10,20]
%%
[datamat,labelvec,subvec] = celldata2mat(feat,cate_feat);
subplot(2,2,1)
boxplot(datamat(1,:),labelvec);
ylabel('bpm')
title (feat_names{1})
[datamat,labelvec,subvec] = celldata2mat(feat_std,cate_feat);
subplot(2,2,3)
boxplot(datamat(1,:),labelvec);
ylabel('bpm')
title (feat_names{1})
[datamat,labelvec,subvec] = celldata2mat(feat_tanh,cate_feat);
subplot(2,2,4)
boxplot(datamat(1,:),labelvec);
ylabel('bpm')
title (feat_names{1})






%% ANOVA, USING ORIGINAL DATA
% pre-processing 
clear all
close all
load('ECGfeatures_egdes_100win');
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
% valid = p_anova<0.05; % reject null hypothesis at the bonferroni corrected p value
counts = sum(double(valid),2); % count n.o features that alt hypo holds
abd_feat = counts<=0; % indices of features should be removed
% abd_feat = counts==0; % indices of features should be removed
save('abd_feat.mat','abd_feat');
% 
% for idx = 1:find(abd_feat)
 feat_names{abd_feat}
% end

%% Normalization of data, two techniques  Preparing Data
clear all
close all
load('ECGfeatures_egdes_100win');
load('abd_feat')
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
% save('feat_std.mat','feat_std');
% save('feat_tanh.mat','feat_tanh');

% SVM CLASSIFICATION, RBF,
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
% 
% % % remove excessive features?? 
% Bmat(abd_feat,:) = [];
% Rmat(abd_feat,:) = [];
% Amat(abd_feat,:) = [];
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
CVSVMModel = crossval(SVMModel,'Holdout',0.30);
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

%% PLOT FEATURES in compact way
clear all
close all
load('ECGfeatures_100_47feat_16s');
mu = mean([feat{:}],2);
sig = std([feat{:}],0,2);
for i = 1:numel(feat)
    feat_std{i} = (feat{i} - mu) ./ sig;
end
% tanh normalization
for i = 1:numel(feat)
    feat_tanh{i} = 0.5.*(tanh(0.01.*(feat{i}-mu)./sig)+1);
end
classNames = categories(cate_feat{1});
figure(2)
title('50s window with 5s increment')
% colormap hsv
dispfeat = 8;
for j = 1:numel(cate_feat)
    subplot(8,2,j)
%     plot(feat{j}(8,:))
    for i = 1:numel(classNames)
        hold on
    label = classNames(i);
    idx = find(cate_feat{j} == label);
    plot(idx,feat_std{j}(dispfeat,idx))
    x0=10;
    y0=10;
    width=300;
    height=500;
    set(gcf,'units','points','position',[x0,y0,width,height])
    end
%     ylabel('iA_{LF}');
    ylabel(string(feat_names{dispfeat}))
    hold off
end
legend('Rest','Prepare','Speech','Math','Recovery','Location','southoutside')

%%%%
sequence
