function [features,hrvaxis_interp] = prepro(hrv,ecg,fs,stage)
%M, ECG data, first column, volt; second column, time stamp in milli sec

% Empirical mode decomposition
[imf,~] = emd(hrv);
hrv_detrend= sum(imf,2); %detrend hrv
% plot(dhrv

rr_fs = 4;  %resampling at 4 Hz
hrv_time = locs(2:end)./fs; % N locations and N-1 HRV ??? hrv time axis in seconds
hrvaxis_interp = 1/rr_fs:1/rr_fs:length(ecg)/fs;%60 time axis in seconds after interpolation
%Feature 1 interpolated HRV
hrv_interp = interp1(hrv_time,hrv_detrend,hrvaxis_interp ,'pchip');

subplot(3,2,6)
% hrvaxis = linspace(0,lengthinsec,length(hrv));
plot(hrvaxis_interp,hrv_interp);
xlabel('seconds');
title('RR interval')

figure,
subplot(3,2,1)
plot(hrvaxis_interp,hrv_interp);
xlabel('time (secs)')
title('Detrended and interpolated RR series')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

%Feature 2 Heartrate
heartRate = 60./hrv_interp; %Heartrate (BPM)


%%%%%%%%%% Frequency domain Analysis%%%%%%%%%%%%%%%%
%300s per segment, 25s window minimum
% 7 mins windows or 250 seconds
increment = rr_fs*10;
winlength = rr_fs*250;
noverlap = winlength - increment;
fvector = linspace(0,2,length(hrv_interp)/2);

[spec,faxis,taxis] = spectrogram(hrv_interp,hamming(winlength),noverlap,fvector,rr_fs);
pxx = spec.*conj(spec);
pLF = sum(pxx(find(faxis >0.04 & faxis <0.15),:),1);
pHF = sum(pxx(find(faxis >0.15 & faxis <0.4),:),1);
LFtoHF = pLF./pHF;

%%%%%%%%%%%pwelch window
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*10;
winlength = rr_fs*250;

while winleft + winlength< length(hrv_interp)
sdRR(ii) = std(hrv_interp(winleft:winleft+winlength));
meanRR(ii) = mean(hrv_interp(winleft:winleft+winlength));
winleft = winleft+increment;
ii = ii+1;
end

tstart = 0.5*winlength/rr_fs;
tend = (length(sdRR)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
timeaxis = linspace(tstart,tend,length(sdRR));


subplot(3,2,2)
plot(taxis,[pLF;pHF]);
title('LF and HF power')
legend('LF power','HF power')
xlabel('time (secs)')
ylabel('power (s^2/Hz)')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

subplot(3,2,3)
plot(taxis,LFtoHF);
xlabel('time (secs)')
title('LF/HF ratio')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

% HRV, emd, summing up, interpolation, LSD.

%%%%%%%%%%Time domain Analysis%%%%%%%%%%%%%%%%

% Feature: standard deviation of RR, mean of RR
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*15;
winlength = rr_fs*150;

while winleft + winlength< length(hrv_interp)
sdRR(ii) = std(hrv_interp(winleft:winleft+winlength));
meanRR(ii) = mean(hrv_interp(winleft:winleft+winlength));
winleft = winleft+increment;
ii = ii+1;
end

tstart = 0.5*winlength/rr_fs;
tend = (length(sdRR)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
timeaxis = linspace(tstart,tend,length(sdRR));

subplot(3,2,4)
plot(timeaxis,sdRR);
xlabel('time (secs)')
title('Standard Deviation of RR')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

% x0=10;
% y0=10;
% width=550;
% height=200;
% set(gcf,'units','points','position',[x0,y0,width,height])

subplot(3,2,5)
plot(timeaxis,meanRR);
xlabel('time (secs)')
title('Mean of RR')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

%%%%%%%%%%Non-Linear Analysis%%%%%%%%%%%%%%%%

% Features: Sample Entropy
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*15;
winlength = rr_fs*150;

while winleft + winlength< length(hrv_interp)
SampEn(ii)= sample_entropy(hrv_interp(winleft:winleft+winlength),2,0.2);
winleft = winleft+increment;
ii = ii+1;
end

tstart = 0.5*winlength/rr_fs;
tend = (length(SampEn)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
timeaxis = linspace(tstart,tend,length(SampEn));

subplot(3,2,6)
plot(timeaxis,SampEn);
xlabel('time (secs)')
title('Sample Entropy of RR')
vline(stage,'--k',{'resting','preparation','speech','math','recovery'});

features = {hrv_interp;heartRate;hrv_detrend;sdRR;pLF;pHF;LFtoHF};

end

