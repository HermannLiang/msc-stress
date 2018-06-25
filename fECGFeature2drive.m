function feats = fECGFeature2drive(data,fs)
% Note:
% Input:
% data : M observations x N dimensions ECG signals
% fs : sampling frequency of the ECG signal
% Output:
% ------------------Features in time domain--------------------------------
% avgHR=Average Heart Rate
% avgRR= mean R-R interval distance
% aveHRV= Average Heart Rate Variability
% rMSSD= Root Mean Square Distance of Successive R-R interval
% nn50= Number of R peaks in ECG that differ more than 50 millisecond
% pNN50= percentage NN50
% sdRR= Standard Deviation of R-R series
% sdHR= Standard Deviation of Heart Rate
%
% ------------------Features in nonlinear analysis-------------------------
% SampEn1,SampEn2= Sample Entropy with dim = 1 and 2
%
% ------------------Features in Poincare analysis--------------------------
% SD1: Standard deviation of RR intervals along the axis perpendicular to
%      the line of identity.
% SD2: Standard deviation of RR intervals along the line of identity.
% SD1_SD2: SD1./SD2
%
% ------------------Features in frequency domain---------------------------
% LF: Power from 0.04 Hz to 0.15 Hz 
% HF: Power from 0.15 Hz to 0.40 Hz
% LF_HF: LF./HF

% You can load data
% load('sub_ardui_1.mat')
% data = data(1:2:end); Odd index (e.g. 1,3,5...) -> ECG data

feats = zeros(size(data,1),23);
for loop = 1:size(data,1)
    %% Initialization
    ecg_signal = data(loop,:); %for each observation
    winsize = length(ecg_signal); %length of the signal
    
    %% Pre-process: This part is used to detecting the R peaks
    % Removing lower frequencies
    
                    fresult=fft(ecg_signal);
                %     fresult(1 : round(length(fresult)*5/fs))=0;
                %     fresult(end - round(length(fresult)*5/fs) : end)=0;
                    fresult(1 : 50)=0; 
                    fresult(end - 50: end)=0; %remove the negatvie frequence as well
                    corrected=real(ifft(fresult));

                    figure(1)
                    subplot(3,2,1)
                    plot(ecg_signal);xlim([0,floor(size(data,2)./500)]); 
                    title('unfiltered')
                    subplot(3,2,2)
                    plot(corrected);xlim([0,floor(size(data,2)./500)]);
                    title('filtered')
                % 
%                     figure,
%                     freqaxis = linspace(0,fs/2,floor(winsize/2));
%                     plot(freqaxis,10*log10(abs(fresult(1:floor(winsize/2)))))
%                     xlabel('Frequency (Hz)');
%                     ylabel('Log Magnitude')

                    % Filter - first pass
                    WinSize = floor(fs * 700 / 1000);% You may change 571 to a proper val
                    % to make sure the window size is odd
                    if rem(WinSize,2)==0
                        WinSize = WinSize+1;
                    end
                    filtered1=ecgdemowinmax(corrected, WinSize);

            %Visualization
            subplot(3,2,3)
            plot(filtered1);xlim([0,floor(size(data,2)./500)]);
    
    % Scale an ecg and a threshold filter
    peaks1=filtered1/max(filtered1);
    ave1 = mean(peaks1(:,peaks1>0));
    std1 = std(peaks1(:,peaks1>0));
    peaks1(:,peaks1<ave1+std1*0.02)= 0; % Threshold is mean-0.1*std
    positions=find(peaks1);
    
%     % Visualization
    subplot(3,2,4)
    plot(peaks1);xlim([0,floor(size(data,2)./500)]);
    
    % Returns minimum distance between two peaks
    distance = min(diff(positions));
    
    % Optimize filter window size
    QRdistance=floor(0.04*fs);
    if rem(QRdistance,2)==0
        QRdistance=QRdistance+1;
    end
    WinSize=2*distance-QRdistance;
    
    % Filter - second pass
    peaks2=ecgdemowinmax(corrected, WinSize);
    
    subplot(3,2,5)
    plot(peaks2);xlim([0,floor(size(data,2)./500)]);
    % Visualization
%     figure(2)
% %     plot(ecg_signal);xlim([0,floor(size(data,2)./10)]); 
%     plot((ecg_signal-min(ecg_signal))/(max(ecg_signal)-min(ecg_signal)), '-g'); hold on
%     stem(peaks2/2);xlim([0,floor(size(data,2)./10)]);
%     title('\bf Deteceted R peaks')
%     stem(peaks2);
%     title('\bf6. Detected Peaks - Finally');  xlim([0,floor(size(data,2)./10)]);
    %% Time Series Features
    locs=find(peaks2);
    % Number of R-peaks
    num_R_peaks=length(locs);
    % Feature 1: Calculating Average Heart Rate
    time_in_sec=(winsize/fs);
    avgHR=(num_R_peaks/time_in_sec)*60;
    
    hrv = diff(locs)./fs; % Heart Rate Variability (HRV)in seconds
    rr_fs = fs; % 4 Hz
    hrv_time = locs(2:end)./fs; % N locations and N-1 HRV ???
    hrv_time_interp = 1/rr_fs:1/rr_fs:length(ecg_signal)/fs;%60
    hrv_interp = interp1(hrv_time,hrv,hrv_time_interp,'pchip');
    heartRate = 60./hrv_interp;
%     figure;
%     plot(hrv_time_interp,hrv_interp);
    
    % Feature 2: Extracting Mean R-R interval
    avgRR=sum(hrv_interp)/length(hrv_interp);
    
    % Feature 3: Average HRV Calculation
    aveHRV=mean(1./hrv_interp);
    
    % Feature 4: Extracting Root Mean Square of the differences of successive
    % R-R interval (RMSSD)
    square_hrv=diff(hrv_interp).^2;
    avg_square_dstnce=sum(square_hrv)/length(square_hrv);
    rMSSD =sqrt(avg_square_dstnce);
    
    % Feature 5: Extracting number of consecutive R-R intervals that differ
    % more than 50 ms
    nn50 = sum(abs(diff(hrv_interp))>0.05);% More than 50ms
    
    % Feature 6: Extracting Percentage value of total consecutive RR interval that
    %differ more than 50ms
    pNN50 = ((nn50/length(hrv_interp))*100);
    
    % Feature 7: Extracting Standard Deviation of RR interval series
    sdRR =std(hrv_interp);
    
    % Feature 8: Extracting Standard Dviation of Heart Rate
    sdHR= std(heartRate);
    
    %% Nonlinear Features
    % Feature 9 : Calculating Sample Entropy
%     SampEn1= SampEn( hrv,1,0.2 );
%     SampEn2= SampEn( hrv,2,0.2 );
    SampEn1= sample_entropy( hrv_interp,1,0.2 );
    SampEn2= sample_entropy( hrv_interp,2,0.2 );
    
    %% Poincare
    [SD1,SD2]=poincare(hrv_interp);
    SD2_SD1 = SD2./SD1;
    
    %% Frequency domain
%     win = hanning(floor(length(hrv)./4));
    [pxx,f]=pwelch(hrv_interp,[],[],[],rr_fs);
%     figure
%     plot(f,10*log(pxx))
    pLF = max(pxx(find(f<0.15 & f>0.04)));
    pHF = max(pxx(find(f>0.15 & f<0.4)));
    LF = sum(pxx(find(f<0.15 & f>0.04)));
    HF = sum(pxx(find(f>0.15 & f<0.4)));
    LF_HF = LF./HF;
    
    %% MSE
    MSE = multiScaleEntropy(hrv_interp,5);
    
%     feats(loop,:) = [avgHR,avgRR,aveHRV,rMSSD,nn50,pNN50,sdRR,sdHR,...
%         SampEn1,SampEn2,...
%         SD1,SD2,SD2_SD1,...
%         pLF,pHF,LF,HF,LF_HF];
    feats(loop,:) = [avgHR,avgRR,sdHR,sdRR,aveHRV,rMSSD,nn50,pNN50,...
        pLF,pHF,LF,HF,LF_HF,...
        SD1,SD2,SD2_SD1,...
        SampEn1,SampEn2,...
        MSE];
%     feats(loop,:) = [avgHR,avgRR,aveHRV,rMSSD,pNN50,sdRR,sdHR,...
%         SampEn1,SampEn2,...
%         SD2,SD2_SD1];
%     feats(loop,:) = [pNN50,...
%         SampEn1,SampEn2,...
%         SD2,SD2_SD1,LF_HF];

end
end
