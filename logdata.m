%%
clear
close all
%0. testing signal
M = dlmread('test04.txt');
[restfeat,restaxis] = prepro(M);
clear M

%%
% combined separed data??

clear
close all
par_no = 4;  %no. of subject

M1 = dlmread(['rest' num2str(par_no) '.txt']);
M2 = dlmread(['pre' num2str(par_no) '.txt']);
M3 = dlmread(['spe' num2str(par_no) '.txt']);
M4 = dlmread(['math' num2str(par_no) '.txt']);
M5 = dlmread(['aft' num2str(par_no) '.txt']);

%remove the first 5 and last 5data, in case of async.

M1(1:5,:) = []; M1(end-5:end,:) = [];
M2(1:5,:) = []; M2(end-5:end,:) = [];
M3(1:5,:) = []; M3(end-5:end,:) = [];
M4(1:5,:) = []; M4(end-5:end,:) = [];
M5(1:5,:) = []; M5(end-5:end,:) = [];

% time stamped normalized 
temp1 = M1(:,2) - M1(1,2)*ones(length(M1),1);
temp2 = M2(:,2) - M2(1,2)*ones(length(M2),1) + temp1(end,1)*ones(length(M2),1);
temp3 = M3(:,2) - M3(1,2)*ones(length(M3),1) + temp2(end,1)*ones(length(M3),1);
temp4 = M4(:,2) - M4(1,2)*ones(length(M4),1) + temp3(end,1)*ones(length(M4),1);
temp5 = M5(:,2) - M5(1,2)*ones(length(M5),1) + temp4(end,1)*ones(length(M5),1);

timestamp = [temp1;temp2;temp3;temp4;temp5];
fT = mean(diff(timestamp));
fs = round(1000/fT);
stage = [0,temp1(end)/1000,temp2(end)/1000,temp3(end)/1000,temp4(end)/1000];
ecg = [M1(:,1);M2(:,1);M3(:,1);M4(:,1);M5(:,1)];
[features,restaxis] = prepro(ecg,fs,stage);


%%
M1(1:5,:) = []; M1(end-5:end,:) = [];
M2(1:5,:) = []; M2(end-5:end,:) = [];
M3(1:5,:) = []; M3(end-5:end,:) = [];
M5(1:5,:) = []; M5(end-5:end,:) = [];

fT = mean(diff(M5(:,2)));
fs = round(1000/fT);


%%
clear
close all
%1. resting stage
M = dlmread('rest04.txt');
[restfeat,resttime,restaxis] = prepro(M);
clear M

%2. preparing stage
M = dlmread('pre04.txt');
[prefeat,pretime,preaxis] = prepro(M);
clear M

%3. speech stage
M = dlmread('spe04.txt');
[spefeat,spetime,speaxis] = prepro(M);
clear M

%4. math  stage
M = dlmread('math04.txt');
[mathfeat,mathtime,mathaxis] = prepro(M);
clear M

%5. recover stage

M = dlmread('aft04.txt');
[aftfeat,afttime,aftaxis] = prepro(M);
clear M

% HRV figure
figure,
subplot(1,5,1);
plot(restaxis,restfeat{1});
ylim([0 1.5]);
title('resting stage');
xlabel('seconds');
ylabel('R-R interval (seconds)');

subplot(1,5,2);
plot(preaxis,prefeat{1});
ylim([0 1.5]);
title('preparing stage')
xlabel('seconds');

subplot(1,5,3);
plot(speaxis,spefeat{1});
ylim([0 1.5]);
title('speech stage')
xlabel('seconds');

subplot(1,5,4);
plot(mathaxis,mathfeat{1});
ylim([0 1.5]);
title('math stage')
xlabel('seconds');

subplot(1,5,5);
plot(aftaxis,aftfeat{1});
ylim([0 1.5]);
title('recovering stage')
xlabel('seconds');
% MSE = multiScaleEntropy(afthrv,5);

% Heartrate figure
figure,
subplot(1,5,1);
plot(restaxis,restfeat{2});
ylim([60 160]);
title('resting stage');
xlabel('seconds');
ylabel('Heartrate(bpm)');

subplot(1,5,2);
plot(preaxis,prefeat{2});
ylim([60 160]);
title('preparing stage')
xlabel('seconds');

subplot(1,5,3);
plot(speaxis,spefeat{2});
ylim([60 160]);
title('speech stage')
xlabel('seconds');

subplot(1,5,4);
plot(mathaxis,mathfeat{2});
ylim([60 160]);
title('math stage')
xlabel('seconds');

subplot(1,5,5);
plot(aftaxis,aftfeat{2});
ylim([60 160]);
title('recovering stage')
xlabel('seconds');
% MSE = multiScaleEntropy(afthrv,5);




% Detrend HRV figure
figure,
subplot(1,5,1);
plot(restaxis,restfeat{3});
ylim([-0.5 0.5]);
title('resting stage');
xlabel('seconds');
ylabel('R-R interval (seconds)');

subplot(1,5,2);
plot(preaxis,prefeat{3});
ylim([-0.5 0.5]);
title('preparing stage')
xlabel('seconds');

subplot(1,5,3);
plot(speaxis,spefeat{3});
ylim([-0.5 0.5]);
title('speech stage')
xlabel('seconds');

subplot(1,5,4);
plot(mathaxis,mathfeat{3});
ylim([-0.5 0.5]);
title('math stage')
xlabel('seconds');

subplot(1,5,5);
plot(aftaxis,aftfeat{3});
ylim([-0.5 0.5]);
title('recovering stage')
xlabel('seconds');


%%
[imf,residual] = emd(aftfeat{1});
