function rr_data = fRRdetection(data,fs)
% Note:
% Input:
% data : M observations x N dimensions ECG signals
% fs : sampling frequency of the ECG signal
% Output:

rr_data = zeros(size(data,1),size(data,2));
for loop = 1:size(data,1)
    %% Initialization
    ecg_signal = data(loop,:);
    winsize = length(ecg_signal);
    
    %% Pre-process: This part is used to detecting the R peaks
    % Removing lower frequencies
    fresult=fft(ecg_signal);
%     fresult(1 : round(length(fresult)*5/fs))=0;
%     fresult(end - round(length(fresult)*5/fs) : end)=0;
    fresult(1 : 50)=0; % You may change 50 to a proper val.
    fresult(end - 50: end)=0;
    corrected=real(ifft(fresult));
    
%     % Visualization
%     figure
%     subplot(3,2,1)
%     plot(ecg_signal);xlim([0,floor(size(data,2)./10)]);
%     subplot(3,2,2)
%     plot(corrected);xlim([0,floor(size(data,2)./10)]);
%     
    % Filter - first pass
    WinSize = floor(fs * 700 / 1000);% You may change 571 to a proper val
    if rem(WinSize,2)==0
        WinSize = WinSize+1;
    end
    filtered1=ecgdemowinmax(corrected, WinSize);
    
% %     Visualization
%     subplot(3,2,3)
%     plot(filtered1);xlim([0,floor(size(data,2)./10)]);
    
    % Scale an ecg and a threshold filter
    peaks1=filtered1/max(filtered1);
    ave1 = mean(peaks1(:,peaks1>0));
    std1 = std(peaks1(:,peaks1>0));
    peaks1(:,peaks1<ave1+std1*0.02)= 0; % Threshold is mean-0.1*std
    positions=find(peaks1);
    
%     % Visualization
%     subplot(3,2,4)
%     plot(peaks1);xlim([0,floor(size(data,2)./10)]);
    
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
    
%     % Visualization
%     subplot(3,2,5)
%     plot(peaks2);xlim([0,floor(size(data,2)./10)]);
    
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
    hrv_time_interp = 1/rr_fs:1/rr_fs:(size(data,2)./fs);
    rr_data(loop,:) = interp1(hrv_time,hrv,hrv_time_interp,'pchip');
end