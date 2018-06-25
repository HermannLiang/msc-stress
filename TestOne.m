%
%Chulin Liang
%% clear all
% try to figure out feature extraction
clear all
load am_s2.mat;
data = data{1};
fs = 50;
% data = data(1:2:end); %Odd index (e.g. 1,3,5...) -> ECG data
%%
 feats = fECGFeature2(data,fs);
% %%
% plot(ecg_signal(60:360));
%% NEW TESTING ECGDEMO
clear all
% load ecgdemodata1; %the demodata from ecgdemo
% load amusement data from youqian
load am_s2.mat;
ecg = data{1};
samplingrate = 50;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The following code is extracted from ecgdemo from librow
    %   Remove lower frequencies
    fresult=fft(ecg);
    fresult(1 : round(length(fresult)*5/samplingrate))=0;
    fresult(end - round(length(fresult)*5/samplingrate) : end)=0;
    corrected=real(ifft(fresult));
    %   Filter - first pass
    WinSize = floor(samplingrate * 700 / 1000);
    if rem(WinSize,2)==0
        WinSize = WinSize+1;
    end
    filtered1=ecgdemowinmax(corrected, WinSize);
    %   Scale ecg
    peaks1=filtered1/(max(filtered1)/7);
    %   Filter by threshold filter
    for data = 1:1:length(peaks1)
        if peaks1(data) < 4
            peaks1(data) = 0;
        else
            peaks1(data)=1;
        end
    end
    positions=find(peaks1);
    distance=positions(2)-positions(1);
    for data=1:1:length(positions)-1
        if positions(data+1)-positions(data)<distance
            distance=positions(data+1)-positions(data);
        end
    end
    % Optimize filter window size
    QRdistance=floor(0.04*samplingrate);
    if rem(QRdistance,2)==0
        QRdistance=QRdistance+1;
    end
    WinSize=2*distance-QRdistance;
    % Filter - second pass
    filtered2=ecgdemowinmax(corrected, WinSize);
    peaks2=filtered2;
    for data=1:1:length(peaks2)
        if peaks2(data)<0.5
            peaks2(data)=0;
        else
            peaks2(data)=1;
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%visualisation
%
%   %   Create figure - stages of processing
%     figure(demo); set(demo, 'Name', strcat(plotname, ' - Processing Stages'));
    %   Original input ECG data
    figure(1)
    subplot(3, 2, 1); plot((ecg-min(ecg))/(max(ecg)-min(ecg)));
    title('\bf1. Original ECG'); ylim([-0.2 1.2]);xlim([0,floor(size(ecg,2)./20)]);
    %   ECG with removed low-frequency component
    subplot(3, 2, 2); plot((corrected-min(corrected))/(max(corrected)-min(corrected)));
    title('\bf2. FFT Filtered ECG'); ylim([-0.2 1.2]);xlim([0,floor(size(ecg,2)./20)]);
    %   Filtered ECG (1-st pass) - filter has default window size
    subplot(3, 2, 3); stem((filtered1-min(filtered1))/(max(filtered1)-min(filtered1)));
    title('\bf3. Filtered ECG - 1^{st} Pass'); ylim([0 1.4]);xlim([0,floor(size(ecg,2)./20)]);
    %   Detected peaks in filtered ECG
    subplot(3, 2, 4); stem(peaks1);
    title('\bf4. Detected Peaks'); ylim([0 1.4]);xlim([0,floor(size(ecg,2)./20)]);
    %   Filtered ECG (2-d pass) - now filter has optimized window size
    subplot(3, 2, 5); stem((filtered2-min(filtered2))/(max(filtered2)-min(filtered2)));
    title('\bf5. Filtered ECG - 2^d Pass'); ylim([0 1.4]);xlim([0,floor(size(ecg,2)./20)]);
    %   Detected peaks - final result
    subplot(3, 2, 6); stem(peaks2);
    title('\bf6. Detected Peaks - Finally'); ylim([0 1.4]);xlim([0,floor(size(ecg,2)./20)]);
    %   Create figure - result
    figure(2)
    %   Plotting ECG in green
    plot((ecg-min(ecg))/(max(ecg)-min(ecg)), '-g'); title('\bf Comparative ECG R-Peak Detection Plot');
    xlim([0,floor(size(ecg,2)./20)]);
    %   Show peaks in the same picture
    hold on
    %   Stemming peaks in dashed black
    stem(peaks2'.*((ecg-min(ecg))/(max(ecg)-min(ecg)))', ':k');
    xlim([0,floor(size(ecg,2)./20)]);
    %   Hold off the figure
    hold off

    stem(peaks2); xlim([3000 6000])
    
