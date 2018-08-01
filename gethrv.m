function [hrv,hrvaxis] = gethrv(subjectno,visual,filtermethod)

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

rr_fs = 4;
% new method
ecg_all = [M1(:,1);M2(:,1);M3(:,1);M4(:,1);M5(:,1)]; 
fs = 200;
originalecgaxis = timestamp./1000;
ecgnewaxis = 1/fs:1/fs:originalecgaxis(end);
ecg = interp1(originalecgaxis,ecg_all,ecgnewaxis','pchip');


tst = 1400;
ted = 1450;

N =length(ecg);  %length of the record
timeaxis = (1:N)/fs;

            if visual == 1
            figure,
            subplot(3,2,1)
            plot(timeaxis(tst*fs:ted*fs),ecg(tst*fs:ted*fs,1));
            title('Original ECG')
            xlabel('seconds')
            end

psd =fft(ecg);

            if visual == 1
            faxis = linspace(0,fs/2,floor(N/2));
            subplot(3,2,2)
            plot(faxis,10*log10(abs(psd(1:floor(N/2)))))
            xlabel('Hz')
            ylabel('Log magnitude')
            title('Spectrum with 50 Hz noise')
            end

% % remove the 50Hz noise, cutoff at 25Hz cutoff = floor(N*(25/(fs/2))/2);
% psd(N/2-(N/2-cutoff):N/2+(N/2-cutoff)) = 0;

%remove the low frequency components
psd(1:25) = 0;
psd(end-25:end) = 0;

%             if visual == 1 subplot(3,2,3)
%             plot(faxis,10*log10(abs(psd(1:floor(N/2))))) xlabel('Hz')
%             ylabel('Log magnitude') title('Spectrum after filtering') end

%ifft
ecg = real(ifft(psd));
%detrend ??
ecg = detrend(ecg,'constant');

            if visual == 1
            subplot(3,2,3)
            plot(timeaxis,ecg);
            xlim([tst, ted]);
            title('filtered ecg')
            xlabel('seconds')
            end
%R peaks detection

%%%%%%%%%%%%%%ONLY ONE FILTERING%%%%%%%%%%%%%%
% Filter - first pass
WinSize = floor(fs * 570 / 1000);% You may change 571 to a proper val
% to make sure the window size is odd
if rem(WinSize,2)==0
    WinSize = WinSize+1;
end
filtered1=ecgdemowinmax(ecg, WinSize);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%Full filtering from librow%%%%%%%%%%%%%%%
%Scale an ecg and a threshold filter
peaks1=filtered1/max(filtered1);
ave1 = mean(peaks1(:,peaks1>0));
std1 = std(peaks1(:,peaks1>0));
peaks1(:,peaks1<ave1+std1*0.02)= 0; % Threshold is mean-0.1*std
positions=find(peaks1);

% Returns minimum distance between two peaks
distance = min(diff(positions));

% Optimize filter window size
QRdistance=floor(0.04*fs);
if rem(QRdistance,2)==0
    QRdistance=QRdistance+1;
end
WinSize=2*distance-QRdistance;
% Filter - second pass
filtered2=ecgdemowinmax(ecg, WinSize);
peaks2=filtered2;
%just passing the vector
filtered1 = peaks2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            %Visualization
            if visual == 1
            subplot(3,2,4)
            plot(timeaxis,ecg); hold on
            
            plot(timeaxis,filtered1);
            xlim([tst, ted]);
            xlabel('seconds');
            title('Peak detection')
            end

locs=find(filtered1);
% Number of R-peaks
num_R_peaks=length(locs);
time_in_sec=(N/fs);
% avgHR=(num_R_peaks/time_in_sec)*60;
% Heart Rate Variability (HRV)in seconds
noisyhrv = diff(locs)./fs;

%%%%%%%%%%%%%%FILTERING PART%%%%%%%%%%%%
if filtermethod == 0
    hrv = noisyhrv;
    if visual == 1
        subplot(3,2,5)
        plot(noisyhrv);
        xlabel('seconds');
        title('Noisy HRV')
    end
end

if filtermethod == 1
% first pass, exclude the points that def not RR points
index = (noisyhrv > 1.1)|(noisyhrv<0.4);
 % a window excluded the abnormal points
win_excluded = [noisyhrv(find(index)-4);noisyhrv(find(index)-3);noisyhrv(find(index)-2);...
    noisyhrv(find(index)-1);noisyhrv(find(index)+1);noisyhrv(find(index)+2);...
    noisyhrv(find(index)+3);noisyhrv(find(index)+4)];
            if visual == 1
            subplot(3,2,5)
            plot(noisyhrv);
            hold on
            plot(find(index),noisyhrv(index),'+'); 
            xlabel('seconds');
            title('Noisy HRV')
            end
            noisyhrv(index) = median(win_excluded,1); % or mean
% second pass, use global mean and std.
mu = mean(noisyhrv);
sigma = std(noisyhrv);
index1 = (noisyhrv > mu+ 3*sigma)|(noisyhrv<mu-3*sigma);  %locate the abnormal RR
if visual == 1
plot(find(index1),noisyhrv(index1),'+'); 
end
%  a window excluded the abnormal point, and there neighbouring points??
win_excluded1 = [noisyhrv(find(index1)-4);noisyhrv(find(index1)-3);noisyhrv(find(index1)-2);...
    noisyhrv(find(index1)-1);noisyhrv(find(index1)+1);noisyhrv(find(index1)+2);...
    noisyhrv(find(index1)+3);noisyhrv(find(index1)+4)];
noisyhrv(index1) = mean(win_excluded1,1); % or mean

% third pass, use local mean and std, adapted from physionet: pp19
% https://www.physionet.org/events/hrv-2006/mietus-1.pdf
prefiltered = noisyhrv; %this variable just for visualisation
winleft = 1; %starting postion of the window
winl = 10;
index2 = false(1,length(noisyhrv)); % logical array
while winleft + winl < length(noisyhrv)
    localwin = noisyhrv(winleft:winleft+winl);
    localwin(1+winl/2) = [];% exclude the middle point
    localmu = mean(localwin);
    localstd = std( localwin);
% if  (noisyhrv(winleft+winl/2) > localmu +3*localstd)||
% (noisyhrv(winleft+winl/2)< localmu - 3*localstd)
if  (noisyhrv(winleft+winl/2) > 1.2 * localmu)|| (noisyhrv(winleft+winl/2)< 0.8*localmu)
    index2(winleft+winl/2) = 1;
    noisyhrv(winleft+winl/2) = localmu;
end
%     indexlocal = (localwin> localmu +3*localstd) | (localwin< localmu -
%     3*localstd); localwin(indexlocal) = localmu;
%     noisyhrv(winleft:winleft+5) = localwin;
    winleft = winleft+ 1;
end
if visual == 1
plot(find(index2),prefiltered(index2),'+'); hold off
end
hrv = noisyhrv;
end



%ultimate technique, use it if you are in despair
if filtermethod ==2
    hrv = medfilt1(noisyhrv,3);
    if visual == 1
        subplot(3,2,5)
        plot(noisyhrv);
        xlabel('seconds');
        title('Noisy HRV')
    end
end

% visualize the denoised hrv
if visual == 1
    subplot(3,2,6)
    plot(hrv);
    xlabel('seconds');
    title('Denoised HRV')
end

hrv_time = locs(2:end)./fs; % N locations and N-1 HRV ??? hrv time axis in seconds
hrvaxis_interp = 1/rr_fs:1/rr_fs:length(ecg)/fs;%60 time axis in seconds after interpolation
hrv_interp = interp1(hrv_time,hrv,hrvaxis_interp ,'pchip');

% and the new hrv should be 
hrv = hrv_interp*1000; %hrv in ms
hrvaxis = 0.25:0.25:length(hrv)/rr_fs;

end

