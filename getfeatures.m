function [feat,feat_names,cate_feat,hrv_s_scale,cate_hrv] = getfeatures(subjectno,para,vis)

% imds = imageDatastore('file*.jpg'); wildcard operators;

visual1 = vis; %visualisation??
M1 = dlmread(['base' num2str(subjectno) '.txt']);
M2 = dlmread(['pre' num2str(subjectno) '.txt']);
M3 = dlmread(['spe' num2str(subjectno) '.txt']);
M4 = dlmread(['math' num2str(subjectno) '.txt']);
M5 = dlmread(['rest' num2str(subjectno) '.txt']);

%remove the first 5 and last 5data, in case of async.

M1(1:5,:) = []; M1(end-5:end,:) = [];
M2(1:5,:) = []; M2(end-5:end,:) = [];
M3(1:5,:) = []; M3(end-5:end,:) = [];
M4(1:5,:) = []; M4(end-5:end,:) = [];
M5(1:5,:) = []; M5(end-5:end,:) = [];

% time stamped alignment
temp1 = M1(:,2) - M1(1,2)*ones(length(M1),1);
temp2 = M2(:,2) - M2(1,2)*ones(length(M2),1) + (temp1(end,1)+20)*ones(length(M2),1);
temp3 = M3(:,2) - M3(1,2)*ones(length(M3),1) + (temp2(end,1)+20)*ones(length(M3),1);
temp4 = M4(:,2) - M4(1,2)*ones(length(M4),1) + (temp3(end,1)+20)*ones(length(M4),1);
temp5 = M5(:,2) - M5(1,2)*ones(length(M5),1) + (temp4(end,1)+20)*ones(length(M5),1);

timestamp = [temp1;temp2;temp3;temp4;temp5];

% the fs was computed from averaged period, wrong? 
fT = mean(diff(timestamp));
fs = round(1000/fT);
rr_fs = 4;
stage = [0,temp1(end)/1000,temp2(end)/1000,temp3(end)/1000,temp4(end)/1000];
ecg_all = {M1(:,1),M2(:,1),M3(:,1),M4(:,1),M5(:,1)}; % ecg of all 5 stages


% new method

ecg_alll = [M1(:,1);M2(:,1);M3(:,1);M4(:,1);M5(:,1)]; 
new_fs = 200;
originalecgaxis = timestamp./1000;
ecgnewaxis = 1/new_fs:1/new_fs:originalecgaxis(end);
ecg_interp = interp1(originalecgaxis,ecg_alll,ecgnewaxis','pchip');

hrv_s_scale = ecg2hrv(ecg_interp,new_fs,rr_fs,visual1); % hrv in seconds

% and the new hrv should be 
hrv = hrv_s_scale*1000; %hrv in ms
hrvaxis = 0.25:0.25:length(hrv)/rr_fs;

if visual1 ==1
figure,
subplot(3,2,1)
plot(hrvaxis,hrv);
xlabel('time (secs)'); ylabel('RR interval (ms)')
title('Interpolated RR series')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

% Detrend data using empirical mode decomposition
% [imf,~] = emd(hrv);
% hrv_detrend= sum(imf,2); %detrend hrv
% or use the detrend hrv??
hrv_detrend = detrend(hrv);

if visual1 ==1
subplot(3,2,2)
plot(hrvaxis,hrv_detrend);
xlabel('time (secs)');ylabel('RR interval (ms)');
title('Detrended and interpolated RR series')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

heartrate = 60*1000./hrv; %heartrate in bpm
diff_rr = diff(hrv);
square_rr = diff_rr.^2;

% Time domain feature, using sliding window
disp('processing time domain features')
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.T_incre;
winlength = rr_fs*para.T_winl;

while winleft + winlength< length(hrv)
    %Feature 1: Mean of Heart Rate
    meanHR(ii) = mean(heartrate(winleft:winleft+winlength));
    %Feature 2: Standard deviation of Heart Rate
    sdHR(ii) = std(heartrate(winleft:winleft+winlength));
    %Feature 3: Mean of RR-intervals
    meanRR(ii) = mean(hrv(winleft:winleft+winlength));
    %Feature 4: Standard deviation of RR-intervals
    sdRR(ii) = std(hrv(winleft:winleft+winlength));
    %Feature 5: Root Mean Square of the differences of successive
    % R-R interval (RMSSD)
    RMSSD(ii) = sqrt(mean(square_rr(winleft:winleft+winlength)));
    %Feature 6: Number of consecutive R-R intervals that differ
    % more than 50 ms
    NN50(ii) = sum(abs(diff_rr(winleft:winleft+winlength))>50);
    %Feature 7: Percentage value of total consecutive RR interval that
    %differ more than 50ms
    pNN50(ii) = 100*NN50(ii)/length(diff_rr);
    winleft = winleft+increment;
    ii = ii+1;
end

tstart = 0.5*winlength/rr_fs;
tend = (length(sdRR)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
timeaxis = linspace(tstart,tend,length(sdRR));

if visual1 ==1
subplot(3,2,3)
plot(timeaxis,meanHR);
xlabel('time (secs)');ylabel('bpm');
title('Mean of Heart Rate')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

if visual1 ==1
subplot(3,2,4)
plot(timeaxis,sdHR);
xlabel('time (secs)');ylabel('bpm');
title('Standard deviation of Heart Rate')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

if visual1 ==1
subplot(3,2,5)
plot(timeaxis,meanRR);
xlabel('time (secs)');ylabel('RR interval (ms)')
title('Mean of RR-intervals')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

if visual1 ==1
subplot(3,2,6)
plot(timeaxis,sdRR);
xlabel('time (secs)');ylabel('RR interval (ms)')
title('Standard deviation of RR-intervals')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

if visual1 ==1
figure,
subplot(3,1,1)
plot(timeaxis,RMSSD);
xlabel('time (secs)');
title('RMSSD')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

if visual1 ==1
subplot(3,1,2)
plot(timeaxis,NN50);
xlabel('time (secs)');
title('NN50')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

if visual1 ==1
subplot(3,1,3)
plot(timeaxis,pNN50);
xlabel('time (secs)');
title('pNN50')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

%%%%%%%%%% Frequency domain Analysis%%%%%%%%%%%%%%%%
% 
% 
% % spectrogram method
% %  300s per segment, 25s window minimum
% % 7 mins windows or 250 seconds
% 
% increment = rr_fs*para.F_incre;
% winlength = rr_fs*para.F_winl;
% noverlap = winlength - increment;
% fvector = linspace(0,2,length(hrv_detrend)/2);
% 
% [spec,faxis,timeaxis] = spectrogram(hrv_detrend,hamming(winlength),noverlap,fvector,rr_fs);
% % pxx = spec.*conj(spec);
% pxx = abs(spec);
% pLF = sum(pxx(find(faxis >0.04 & faxis <0.15),:),1);
% pHF = sum(pxx(find(faxis >0.15 & faxis <0.4),:),1);
% LFtoHF = pLF./pHF;
% 
% 
% figure,
% subplot(2,1,1)
% plot(timeaxis,[pLF;pHF]);
% title('LF and HF power Spectrogram method')
% legend('LF power','HF power')
% xlabel('time (secs)')
% ylabel('power (s^2/Hz)')
% vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
% 
% subplot(2,1,2)
% plot(timeaxis,LFtoHF);
% xlabel('time (secs)')
% title('LF/HF ratio')
% vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
% 
% clearvars pxx pLf pHf LFtoHF

%%%%%%%%%%%pwelch window
disp('processing frequency domain features')
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.F_incre;
winlength = rr_fs*para.F_winl;
pwelchwin = rr_fs*para.F_pwinl; %pwelch window length
noverlap = rr_fs*para.F_over; %pwelch no overlapping sampling

%%%%%%%%%%%% instantaneous amplitude
hrv_lf = bandpass(hrv_detrend,[0.04 0.15],rr_fs);
hrv_hf = bandpass(hrv_detrend,[0.15 0.4],rr_fs);

ana_lf = hilbert(hrv_lf);
ana_hf = hilbert(hrv_hf);



while winleft + winlength< length(hrv_detrend)
    
    temp_iA_LF = abs(ana_lf(winleft:winleft+winlength)); % obtain the amplitude
    [~,idxmax20LF] = maxk(temp_iA_LF,floor(0.2*winlength));
    [~,idxmin20LF] = mink(temp_iA_LF,floor(0.2*winlength)); % get the indices of max and min 20%
    temp_iA_LF([idxmax20LF, idxmin20LF]) = []; % exclude the max20% and min20%
    iA_LF(ii) = mean(temp_iA_LF);
    
    temp_iA_HF = abs(ana_hf(winleft:winleft+winlength)); % obtain the amplitude
    [~,idxmax20HF] = maxk(temp_iA_HF,floor(0.2*winlength));
    [~,idxmin20HF] = mink(temp_iA_HF,floor(0.2*winlength)); % get the indices of max and min 20%
    temp_iA_HF([idxmax20HF, idxmin20HF]) = []; % exclude the max20% and min20%
    iA_HF(ii) = mean(temp_iA_HF);

%     [pxx,faxis] = pwelch(hrv_detrend(winleft:winleft+winlength),hamming(pwelchwin),noverlap,[],rr_fs);
    [pxx,faxis] = pwelch(hrv_detrend(winleft:winleft+winlength),[],[],[],rr_fs);
% normalization?
%     pLFandHF = sum(pxx(find(faxis >0.04 & faxis <0.5),:),1);
    pLFandHF = 1; % no normalization 
    pLF(ii) = sum(pxx(find(faxis >0.04 & faxis <0.15),:),1)/pLFandHF;
    pHF(ii) = sum(pxx(find(faxis >0.15 & faxis <0.4),:),1)/pLFandHF;
    winleft = winleft+increment;
    ii = ii+1;
end

%HRV spectrum
% figure,
% plot(faxis,10*log10(pxx))
% xlabel('Frequency (Hz)')
% ylabel('ms^2/Hz')

LFtoHF = pLF./pHF;

tstart = 0.5*winlength/rr_fs;
tend = (length(pLF)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
timeaxis = linspace(tstart,tend,length(pLF));

if visual1 ==1

figure,
subplot(2,1,1)
plot(timeaxis,[pLF;pHF]);
title('LF and HF power, Pwelch method')
legend('LF power','HF power')
xlabel('time (secs)')
ylabel('power (s^2/Hz)')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

subplot(2,1,2)
plot(timeaxis,LFtoHF);
xlabel('time (secs)')
title('LF/HF ratio')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

figure,
subplot(2,1,1)
plot(timeaxis,iA_LF);
xlabel('time (secs)')
ylabel('LF_{iA} (ms)')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
title('Instantaneous amplitude (iA) of LF')

subplot(2,1,2)
plot(timeaxis,iA_HF);
xlabel('time (secs)')
ylabel('HF_{iA} (ms)')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
title('Instantaneous amplitude (iA) of HF')
end
%%%%%%%%%%Non-Linear Analysis%%%%%%%%%%%%%%%%
%variation: just use the hrv instead of detrended version
hrv_detrend = hrv;

% Features: Sample Entropy
disp('processing non-linear domain features')
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.N_incre;
winlength = rr_fs*para.N_winl;


while winleft + winlength< length(hrv_detrend)
    temp = sample_entropy(hrv_detrend(winleft:winleft+winlength),2,0.15);
    temp1 = sample_entropy(hrv_detrend(winleft:winleft+winlength),2,0.2);
    temp2 = sample_entropy(hrv_detrend(winleft:winleft+winlength),1,0.2);
    temp3 = sample_entropy(hrv_detrend(winleft:winleft+winlength),1,0.15);
    SampEn1(1,ii)= temp(1);
    SampEn2(1,ii)= temp1(1);
    SampEn3(1,ii)= temp2(1);
    SampEn4(1,ii)= temp3(1);
%     MSE(:,ii) = multiScaleEntropy(hrv_detrend(winleft:winleft+winlength),5);
    MSE(:,ii) = msentropy(hrv_detrend(winleft:winleft+winlength),2,0.15,5);
%     FuzzEnt(ii) = mmfe(hrv_detrend(winleft:winleft+winlength),2,1,0.15,2,5);
    MFE(:,ii) = MFE_mu(hrv_detrend(winleft:winleft+winlength),2,0.15,2,1,5)';
    winleft = winleft+increment;
    ii = ii+1;
end

tstart = 0.5*winlength/rr_fs;
tend = (length(SampEn1)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
SE_timeaxis = linspace(tstart,tend,length(SampEn1));


if visual1 ==1
figure,
subplot(3,1,1)
plot(SE_timeaxis,[SampEn1;SampEn2;SampEn3;SampEn4]);
xlabel('time (secs)')
legend('m = 2, r = 0.15','m = 2, r = 0.2','m = 1, r = 0.2','m = 1, r = 0.15')
title(['Sample Entropy of RR, Subject ', num2str(subjectno)])
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

subplot(3,1,2)
plot(SE_timeaxis,MSE);
xlabel('time (secs)')
title(['MSE, scale factor from 1 to 5, Subject ', num2str(subjectno)])
legend('1','2','3','4','5')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

subplot(3,1,3)
plot(SE_timeaxis,MFE);
xlabel('time (secs)')
title(['MFE, scale factor from 1 to 5, Subject ', num2str(subjectno)])
legend('1','2','3','4','5')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

MSE1 = MSE(1,:);MSE2 = MSE(2,:);MSE3 = MSE(3,:);MSE4 = MSE(4,:);MSE5 = MSE(5,:);
MFE1 = MFE(1,:);MFE2 = MFE(2,:);MFE3 = MFE(3,:);MFE4 = MFE(4,:);MFE5 = MFE(5,:);

% create categorical array
for j = 1:5
[~,edgeidx(j)] = min(abs(timeaxis - ones(1,length(timeaxis))*stage(j)));
[~,edgeidx_hrv(j)] = min(abs(hrvaxis - ones(1,length(hrvaxis))*stage(j)));
end
Y = discretize(timeaxis,[timeaxis(edgeidx),timeaxis(end)]);
Y_hrv = discretize(hrvaxis,[hrvaxis(edgeidx_hrv),hrvaxis(end)]);
cate_hrv = categorical(Y_hrv,[1 2 3 4 5],{'resting','alarm','resistance','resistance','recovery'});
cate_feat = categorical(Y,[1 2 3 4 5],{'resting','alarm','resistance','resistance','recovery'});
categories(cate_feat)

% features_cell
feat = [meanHR;sdHR;meanRR;sdRR;RMSSD;NN50;pNN50;...
    iA_LF;iA_HF;pLF;pHF;LFtoHF;...
    SampEn1;SampEn2;SampEn3;SampEn4;...
    MSE1;MSE2;MSE3;MSE4;MSE5;...
    MFE1;MFE2;MFE3;MFE4;MFE5];

feat_names = {'meanHR','sdHR','meanRR','sdRR','RMSSD','NN50','pNN50',...
    'iA_LF','iA_HF','pLF','pHF','LFtoHF',...
    'SampEn1','SampEn2','SampEn3','SampEn4',...
    'MSE1','MSE2','MSE3','MSE4','MSE5',...
    'MFE1','MFE2','MFE3','MFE4','MFE5'};

end
