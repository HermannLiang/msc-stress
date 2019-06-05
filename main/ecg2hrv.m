function [hrv_interp] = ecg2hrv(ecg,fs,rr_fs,vis)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
visual = vis;
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

% % mean1 = mean(noisyhrv);
% % first pass, exclude the points that def not RR points
% index = (noisyhrv > 1.1)|(noisyhrv<0.4);
%  % a window excluded the abnormal points
% % win_excluded = [noisyhrv(find(index)-4);noisyhrv(find(index)-3);noisyhrv(find(index)-2);...
% %     noisyhrv(find(index)-1);noisyhrv(find(index)+1);noisyhrv(find(index)+2);...
% %     noisyhrv(find(index)+3);noisyhrv(find(index)+4)];

            if visual == 1
            subplot(3,2,5)
            plot(noisyhrv);
%             hold on
%             plot(find(index),noisyhrv(index),'+'); 
            xlabel('seconds');
            title('RR interval')
            end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             noisyhrv(index) = median(win_excluded,1); % or mean
% % M = movmedian(noisyhrv,5,'SamplePoints',double(index));
% % second pass, use global mean and std. 
% mu = mean(noisyhrv);
% sigma = std(noisyhrv);
% index1 = (noisyhrv > mu+ 3*sigma)|(noisyhrv<mu-3*sigma);  %locate the abnormal RR
% if visual == 1
% plot(find(index1),noisyhrv(index1),'+'); 
% end
% %  a window excluded the abnormal point, and there neighbouring points?? 
% win_excluded1 = [noisyhrv(find(index1)-4);noisyhrv(find(index1)-3);noisyhrv(find(index1)-2);...
%     noisyhrv(find(index1)-1);noisyhrv(find(index1)+1);noisyhrv(find(index1)+2);...
%     noisyhrv(find(index1)+3);noisyhrv(find(index1)+4)];
% noisyhrv(index1) = mean(win_excluded1,1); % or mean
% 
% % third pass, use local mean and std, adapted from physionet: pp19
% % https://www.physionet.org/events/hrv-2006/mietus-1.pdf 
% prefiltered = noisyhrv; %this variable just for visualisation
% winleft = 1; %starting postion of the window
% winl = 10;
% index2 = false(1,length(noisyhrv)); % logical array
% while winleft + winl < length(noisyhrv)
%     localwin = noisyhrv(winleft:winleft+winl);
%     localwin(1+winl/2) = [];% exclude the middle point
%     localmu = mean(localwin);
%     localstd = std( localwin);
% % if  (noisyhrv(winleft+winl/2) > localmu +3*localstd)|| (noisyhrv(winleft+winl/2)< localmu - 3*localstd)
% if  (noisyhrv(winleft+winl/2) > 1.2 * localmu)|| (noisyhrv(winleft+winl/2)< 0.8*localmu)
%     index2(winleft+winl/2) = 1;
%     noisyhrv(winleft+winl/2) = localmu;
% end
% %     indexlocal = (localwin> localmu +3*localstd) | (localwin< localmu - 3*localstd);
% %     localwin(indexlocal) = localmu;
% %     noisyhrv(winleft:winleft+5) = localwin;
%     winleft = winleft+ 1;
% end
% if visual == 1
% plot(find(index2),prefiltered(index2),'+'); hold off
% end
% hrv = noisyhrv;
% 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ultimate technique, use it if you are in despair
hrv = medfilt1(noisyhrv,3);
if visual == 1
    subplot(3,2,6)
    plot(hrv);
    xlabel('seconds');
    title('denoised HRV')
end

% or just not denoise it? Seriously ? 
%hrv = noisyhrv;

hrv_time = locs(2:end)./fs; % N locations and N-1 HRV ??? hrv time axis in seconds
hrvaxis_interp = 1/rr_fs:1/rr_fs:length(ecg)/fs;%60 time axis in seconds after interpolation
hrv_interp = interp1(hrv_time,hrv,hrvaxis_interp ,'pchip');


end

