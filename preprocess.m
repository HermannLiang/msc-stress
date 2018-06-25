%New logging data
% filename = 'dataday1.txt'; M = dlmread('dataday1.txt',' ',0,2);

% remove first and the last few lines, incase of baud rate async;
M(1:10,:) = [];
M(end-10:end,:) = [];
fT = mean(diff(M(:,2)));
fs = round(1000/fT);

% time plot interval
tst = 2;
ted = 5;
%data12 the best result
% T = 1;                  %sampling period in ms good result with 10 no
% delay?? now baud rate is 9600 fs = 1000/T;
% fs = 160*12;
M = M(:,1);
N =length(M);  %length of the record
lengthinsec = N/fs;
timeaxis = (1:N)/fs;
figure,
subplot(3,2,1)
plot(timeaxis(tst*fs:ted*fs),M(tst*fs:ted*fs));
title('Original ECG')

psd =fft(M);
faxis = linspace(0,fs/2,floor(N/2));
subplot(3,2,2)
plot(faxis,10*log10(abs(psd(1:floor(N/2)))))
xlabel('Hz')
ylabel('Log magnitude')
title('Spectrum with 50 Hz noise')

% remove the 50Hz noise, cutoff at 25Hz
cutoff = floor(N*(25/(fs/2))/2);
psd(N/2-(N/2-cutoff):N/2+(N/2-cutoff)) = 0;

%remove the low frequency components
psd(1:25) = 0;
psd(end-25:end) = 0;
subplot(3,2,3)
plot(faxis,10*log10(abs(psd(1:floor(N/2)))))
xlabel('Hz')
ylabel('Log magnitude')
title('Spectrum after filtering')

%ifft
ecg = real(ifft(psd));
subplot(3,2,4)
plot(timeaxis(tst*fs:ted*fs),ecg(tst*fs:ted*fs));
title('filtered ecg')

%R peaks detection
% Filter - first pass
WinSize = floor(fs * 700 / 1000);% You may change 571 to a proper val
% to make sure the window size is odd
if rem(WinSize,2)==0
    WinSize = WinSize+1;
end
filtered1=ecgdemowinmax(ecg, WinSize);

%Visualization
subplot(3,2,5)
plot(timeaxis,filtered1);
xlabel('seconds');
title('Peak detection')

locs=find(filtered1);
% Number of R-peaks
num_R_peaks=length(locs);
% Feature 1: Calculating Average Heart Rate
time_in_sec=(N/fs);
avgHR=(num_R_peaks/time_in_sec)*60;

hrv = diff(locs)./fs; % Heart Rate Variability (HRV)in seconds

subplot(3,2,6)
plot(hrv);
title('RR interval')




% Scale an ecg and a threshold filter
% peaks1=filtered1/max(filtered1);
% ave1 = mean(peaks1(:,peaks1>0));
% std1 = std(peaks1(:,peaks1>0));
% peaks1(:,peaks1<ave1+std1*0.02)= 0; % Threshold is mean-0.1*std
% positions=find(peaks1);
% 
% %     % Visualization
% subplot(3,2,4)
% plot(peaks1);
% 
% % Returns minimum distance between two peaks
% distance = min(diff(positions));
% 
% % Optimize filter window size
% QRdistance=floor(0.04*fs);
% if rem(QRdistance,2)==0
%     QRdistance=QRdistance+1;
% end
% WinSize=2*distance-QRdistance;
% 
% % Filter - second pass
% peaks2=ecgdemowinmax(ecg, WinSize);
% 
% subplot(3,2,5)
% plot(peaks2);


