%CNN HRV Classification
%% Preparing data
clear all
close all
delete 'ECGdatabase\imagedatabase\training\*.bmp'
delete 'ECGdatabase\imagedatabase\testing\*.bmp'

addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
imgpath = 'C:\Work\Imperial\Projects\Matlab code\ECGdatabase\imagedatabase';
% load('ECGfeatures_100_28feat');
load('ECGfeatures_100_47feat_16s');

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
%%  Leave one out
for i = 1:numel(cate_feat)
    B = setcats(cate_feat{i},{'baseline','math'});
    feat_3{i} = feat_tanh{i}(:,~isundefined(B));
%remove feature ??
%     feat_3{i}(abd_feat,:) = [];
    cate_feat_3{i} = B(:,~isundefined(B));
end
total = numel(feat);
no_train = 14;
no_test = total - no_train;
div = randperm(total); 
dataTrain = feat_3(div(1:no_train));
labelTrain = cate_feat_3(div(1:no_train));
dataTest = feat_3(div(no_train+1:end));
labelTest = cate_feat_3(div(no_train+1:end));

%% Shuffle method 
[datamat,labelvec,subvec] = celldata2mat(feat_tanh,cate_feat);
idx_m = setcats(labelvec,{'baseline','math'});
feat_m= datamat(:,~isundefined(idx_m));
cate_m = labelvec(:,~isundefined(idx_m));
cate_m = removecats(cate_m);
countcats(cate_m)
categories(cate_m)
subjec_m = subvec(:,~isundefined(idx_m));

%'Shuffled' partition
datasize = size(feat_m,2);
partition_idx = randperm(datasize);
train_idx = partition_idx(1:round(0.8*datasize));
test_idx = partition_idx(round(0.8*datasize)+1:end);
XTrain = feat_m(:,train_idx);
YTrain = cate_m(:,train_idx);
XTest = feat_m(:,test_idx);
YTest = cate_m(:,test_idx);

%         [V,W,D] = pca(XTrain','NumComponents',47);
%         XTrain = W';
%         XTest = V'*(XTest - mean(XTest,2));
%         XTest = V'*(XTest' - mean(XTest',2));
%         dataset = W'; 
%% converting training data into images for shuffled
% 
tic
for i = 1: size(XTrain,2)
        gpuarrayA = XTrain(:,i);
        gpuarrayI = mat2gray(gpuarrayA);
        imwrite(gpuarrayI,['ECGdatabase\imagedatabase\training\' num2str(i) '.bmp'],'bmp');
end
imds_train = imageDatastore('ECGdatabase\imagedatabase\training\*.bmp' );
imds_train.Labels = YTrain;


for i = 1: size(XTest,2)
        gpuarrayA = XTest(:,i);
        gpuarrayI = mat2gray(gpuarrayA);
        imwrite(gpuarrayI,['ECGdatabase\imagedatabase\testing\' num2str(i) '.bmp'],'bmp');
end
toc

imds_test = imageDatastore('ECGdatabase\imagedatabase\testing\*.bmp' );
imds_test.Labels = YTest;
img = readimage(imds_test,size(XTest,2));
figure,
plot(gpuarrayI);
figure,
plot(double(img));

%%
 %Method One from Youqian, 1 img per multivariate sample.
% converting training data into images
tic
numidx = 1;
labels_mat = [];
for i = 1:numel(dataTrain)
    for j = 1:size(dataTrain{i},2)
        gpuarrayA = dataTrain{i}(:,j);
        gpuarrayI = mat2gray(gpuarrayA);
        imwrite(gpuarrayI,['ECGdatabase\imagedatabase\training\' num2str(numidx) '.bmp'],'bmp');
        numidx = numidx +1;
    end
    labels_mat = [labels_mat,labelTrain{i}];
end
imds_train = imageDatastore('ECGdatabase\imagedatabase\training\*.bmp' );
imds_train.Labels = labels_mat;
img = readimage(imds_train,1);
size(img)

% converting testing data into images
numidx = 1;
labels_mat = [];
for i = 1:numel(dataTest)
    for j = 1:size(dataTest{i},2)
        gpuarrayA = dataTest{i}(:,j);
        gpuarrayI = mat2gray(gpuarrayA);
        imwrite(gpuarrayI,['ECGdatabase\imagedatabase\testing\' num2str(numidx) '.bmp'],'bmp');
        numidx = numidx +1;
    end
    labels_mat = [labels_mat,labelTest{i}];
end
toc

imds_test = imageDatastore('ECGdatabase\imagedatabase\testing\*.bmp' );
imds_test.Labels = labels_mat;

%% Load image database 
imds_train = imageDatastore('ECGdatabase\imagedatabase\*.bmp' );
imds_test = imageDatastore('ECGdatabase\imagedatabase\testing\*.bmp' );
%% Training network
imagedim = length(gpuarrayA);
    layers = [imageInputLayer([imagedim 1 1])
        convolution2dLayer([2,1],60)
        reluLayer
        maxPooling2dLayer([2,1],'Stride',1)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer()];
    
    options = trainingOptions('sgdm','MaxEpochs',100, ...
        'InitialLearnRate',0.0001,...
        'Plots','training-progress');
%         'LearnRateDropPeriod',20, ...
        
    net = trainNetwork(imds_train,layers,options);
    
    %%
    [Ypred,scores]=classify(net,imds_test);
    acc = sum(Ypred==imds_test.Labels)/numel(Ypred)
    plotconfusion(imds_test.Labels,Ypred);
    %% Pattern net finally
clear all
close all
addpath(genpath('C:\Work\Imperial\Projects\Matlab code'));
load('ECGfeatures_100_47feat_16s');

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
% grp2idx(cate_m)
% countcats(cate_m)
categories(cate_m)
subjec_m = subvec(:,~isundefined(idx_m));
int_cate = grp2idx(cate_m)';
catenum = numel(categories(cate_m));
target = zeros(catenum,length(cate_m));
for i = 1:catenum
    target(i, int_cate ==i ) = 1;
end

% net = patternnet([47,200,50,100]);
% net = patternnet([47,15,3]);
% acc = zeros(1,200);
% numhiddenlayers = [200 50 100];
numhiddenlayers = 50
    for trial = 1:10
% numhiddenlayers = [200,50,100];
net = patternnet(numhiddenlayers);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;
% net = train(net,feat_m,target);

% Train the Network
[net,tr] = train(net,feat_m,target);
% Test the Network
y = net(feat_m);
e = gsubtract(target,y);
performance = perform(net,target,y);
tvec = target.* tr.testMask{1};
yvec = y.* tr.testMask{1};

tvec(:,isnan(tvec(1,:))) = [];
yvec(:,isnan(yvec(1,:))) = [];
tind = vec2ind(tvec);
yind = vec2ind(yvec);
acc(trial)= sum(tind == yind)/numel(tind);
    end
acc_mat(class_set)= mean(acc);
end
% accmax(class_set) = max(acc);

% end
% plot(acc)
% [v,i] = max(acc)
%%
[x,t] = iris_dataset;
net = patternnet(15);
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y);
classes = vec2ind(y);

%%
figure,

barcat = {'10','20','30','40','50'};
barcat_c = categorical(barcat);
barcat_c = reordercats(barcat_c,barcat);
bar(barcat_c,acc_mat);
xlabel('Number of neurons in the hidden layer')
ylabel('accuracy')