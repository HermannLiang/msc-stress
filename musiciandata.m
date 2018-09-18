% testing theerasak data.
clear
close all
load('Violinist06.mat');
visual1 = 1;
mean(diff(rr_time_unfilt_clean));
%the parameters
para.T_winl = 150;
para.T_incre = 10;

para.F_winl = 250;
para.F_incre = 10;
para.F_pwinl = 100;
para.F_over = 50;

para.N_winl = 200;
para.N_incre = 10;

duration = 60*sum(annotation(1:3));
stage = 60*[0,annotation(1),sum(annotation(1:2))];

rr_fs = 8;
hrvaxis = 1/rr_fs:1/rr_fs:duration;%60 time axis in seconds after interpolation
hrv = interp1(rr_time_unfilt_clean,rr_val_unfilt_clean,hrvaxis ,'pchip');
%%
% [feat{1,subjectno},feat_names,cate_feat{1,subjectno},cate_hrv{1,subjectno}] = getfeatures(subjectno,...
%     hrv{subjectno},stage{subjectno},para,vis);

% hrvaxis = 0.25:0.25:length(hrv)/rr_fs;

if visual1 ==1
figure,
subplot(3,2,1)
plot(hrvaxis,hrv);
xlabel('time (secs)'); ylabel('RR interval (ms)')
title('Interpolated RR series')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
end

% Detrend data using empirical mode decomposition
[imf,~] = emd(hrv);
hrv_detrend= sum(imf,2); %detrend hrv

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
    pNN50(ii) = 100* NN50(ii)/length(diff_rr);
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
plot(timeaxis,meanHR);
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
% spectrogram method
%  300s per segment, 25s window minimum
% 7 mins windows or 250 seconds

increment = rr_fs*para.F_incre;
winlength = rr_fs*para.F_winl;
noverlap = winlength - increment;
fvector = linspace(0,2,length(hrv_detrend)/2);

[spec,faxis,timeaxis] = spectrogram(hrv_detrend,hamming(winlength),noverlap,fvector,rr_fs);
pxx = abs(spec);
pLF = sum(pxx(find(faxis >0.04 & faxis <0.15),:),1);
pHF = sum(pxx(find(faxis >0.15 & faxis <0.4),:),1);
LFtoHF = pLF./pHF;


figure,
subplot(2,1,1)
plot(timeaxis,[pLF;pHF]);
title('LF and HF power Spectrogram method')
legend('LF power','HF power')
xlabel('time (secs)')
ylabel('power (s^2/Hz)')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

subplot(2,1,2)
plot(timeaxis,LFtoHF);
xlabel('time (secs)')
title('LF/HF ratio')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

clearvars pxx pLf pHf LFtoHF

%%%%%%%%%%%pwelch window

winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.F_incre;
winlength = rr_fs*para.F_winl;
pwelchwin = rr_fs*para.F_pwinl; %pwelch window length
noverlap = rr_fs*para.F_over; %pwelch no overlapping sampling

while winleft + winlength< length(hrv_detrend)
    
    [pxx,faxis] = pwelch(hrv_detrend(winleft:winleft+winlength),hamming(pwelchwin),noverlap,[],rr_fs);
%     pLFandHF = sum(pxx(find(faxis >0.04 & faxis <0.5),:),1);
    pLFandHF = 1;
    pLF(ii) = sum(pxx(find(faxis >0.04 & faxis <0.15),:),1)/pLFandHF;
    pHF(ii) = sum(pxx(find(faxis >0.15 & faxis <0.4),:),1)/pLFandHF;
    winleft = winleft+increment;
    ii = ii+1;
end

LFtoHF = pLF./pHF;

tstart = 0.5*winlength/rr_fs;
tend = (length(pLF)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
timeaxis = linspace(tstart,tend,length(pLF));

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

%%%%%%%%%%Non-Linear Analysis%%%%%%%%%%%%%%%%

% Features: Sample Entropy
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.N_incre;
winlength = rr_fs*para.N_winl;


while winleft + winlength< length(hrv_detrend)
    temp = sample_entropy(hrv_detrend(winleft:winleft+winlength),2,0.15);
    temp1 = sample_entropy(hrv_detrend(winleft:winleft+winlength),2,0.2);
    temp2 = sample_entropy(hrv_detrend(winleft:winleft+winlength),1,0.2);
    temp3 = sample_entropy(hrv_detrend(winleft:winleft+winlength),1,0.15);
    SampEn(1,ii)= temp(1);
    SampEn(2,ii)= temp1(1);
    SampEn(3,ii)= temp2(1);
    SampEn(4,ii)= temp3(1);
    MSE(:,ii) = multiScaleEntropy(hrv_detrend(winleft:winleft+winlength),5);
    winleft = winleft+increment;
    ii = ii+1;
end

tstart = 0.5*winlength/rr_fs;
tend = (length(SampEn)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
SE_timeaxis = linspace(tstart,tend,length(SampEn));


figure,
plot(SE_timeaxis,SampEn);
xlabel('time (secs)')
legend('m = 2, r = 0.15','m = 2, r = 0.2','m = 1, r = 0.2','m = 1, r = 0.15')
title('Sample Entropy of RR')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

figure,
plot(SE_timeaxis,MSE);
xlabel('time (secs)')
title('MSE, scaling factor from 1 to 5')
legend('1','2','3','4','5')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});
;