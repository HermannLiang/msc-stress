function [feat,feat_names,cate_feat,cate_hrv] = getfeatures(subjectno,hrv,stage,para,visual1)
rr_fs = 4;
hrvaxis = 0.25:0.25:length(hrv)/rr_fs;

for j = 1:5
[~,edgeidx_hrv(j)] = min(abs(hrvaxis - ones(1,length(hrvaxis))*stage(j)));
end

if visual1 ==1
figure,
subplot(4,3,1)
plot(hrvaxis,hrv);
xlabel('time (secs)'); ylabel('RR interval (ms)')
title(['RR series of Subject ', num2str(subjectno)])
vline(stage,'--k',{'R1','P','S','M','R2'});
end

hrv_seg = {hrv(1:edgeidx_hrv(2)-1),hrv(edgeidx_hrv(2):edgeidx_hrv(3)-1),hrv(edgeidx_hrv(3):edgeidx_hrv(4)-1),...
    hrv(edgeidx_hrv(4):edgeidx_hrv(5)-1),hrv(edgeidx_hrv(5):end)};

% Detrend data using empirical mode decomposition
% [imf,~] = emd(hrv);
% hrv_detrend= sum(imf,2); %detrend hrv

% hrv_detrend = detrend(hrv);
hrv_detrend_seg = {detrend(hrv(1:edgeidx_hrv(2)-1)),...
    detrend(hrv(edgeidx_hrv(2):edgeidx_hrv(3)-1)),...
    detrend(hrv(edgeidx_hrv(3):edgeidx_hrv(4)-1)),...
    detrend(hrv(edgeidx_hrv(4):edgeidx_hrv(5)-1)),...
    detrend(hrv(edgeidx_hrv(5):end)),...
    };
hrv_detrend = cell2mat(hrv_detrend_seg);


if visual1 ==1
subplot(4,3,2)
plot(hrvaxis,hrv_detrend);
xlabel('time (secs)');ylabel('RR interval (ms)');
title('Detrended and interpolated RR series')
vline(stage,'--k',{'R1','P','S','M','R2'});
end
% Time domain feature, using sliding window
disp('processing time domain features')
for segment = 1:5
heartrate = 60*1000./hrv_seg{segment}; %heartrate in bpm
diff_rr = diff(hrv_seg{segment});
square_rr = diff_rr.^2;

winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.T_incre;
winlength = rr_fs*para.T_winl;

while winleft + winlength< length(hrv_seg{segment})
    %Feature 1: Mean of Heart Rate
    meanHR_{segment}(ii) = mean(heartrate(winleft:winleft+winlength));
    %Feature 2: Standard deviation of Heart Rate
    sdHR_{segment}(ii) = std(heartrate(winleft:winleft+winlength));
    %Feature 3: Mean of RR-intervals
    meanRR_{segment}(ii) = mean(hrv_seg{segment}(winleft:winleft+winlength));
    %Feature 4: Standard deviation of RR-intervals
    sdRR_{segment}(ii) = std(hrv_seg{segment}(winleft:winleft+winlength));
    %Feature 5: Root Mean Square of the differences of successive
    % R-R interval (RMSSD)
    RMSSD_{segment}(ii) = sqrt(mean(square_rr(winleft:winleft+winlength)));
    %Feature 6: Number of consecutive R-R intervals that differ
    % more than 50 ms
    NN50_{segment}(ii) = sum(abs(diff_rr(winleft:winleft+winlength))>50);
    %Feature 7: Percentage value of total consecutive RR interval that
    %differ more than 50ms
    pNN50_{segment}(ii) = 100*NN50_{segment}(ii)/length(diff_rr);
    winleft = winleft+increment;
    ii = ii+1;
end


end

meanHR = cell2mat(meanHR_);
sdHR = cell2mat(sdHR_);
meanRR = cell2mat(meanRR_);
sdRR = cell2mat(sdRR_);
RMSSD = cell2mat(RMSSD_);
NN50 = cell2mat(NN50_);
pNN50 = cell2mat(pNN50_);

tstart = 0.5*winlength/rr_fs;
tend = (length(sdRR)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
timeaxis = linspace(tstart,tend,length(sdRR));
l1 = length(meanRR_{1});
l2 = length(meanRR_{2});
l3 = length(meanRR_{3});
l4 = length(meanRR_{4});
stagenew = [1,l1,l1+l2,l1+l2+l3,l1+l2+l3+l4];

if visual1 ==1
subplot(4,3,3)
plot(meanHR);
xlabel('time step');ylabel('bpm');
title('Mean of Heart Rate')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
end

if visual1 ==1
subplot(4,3,4)
plot(sdHR);
xlabel('time step');ylabel('bpm');
title('Standard deviation of Heart Rate')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
end

if visual1 ==1
subplot(4,3,5)
plot(meanRR);
xlabel('time step');ylabel('RR interval (ms)')
title('Mean of RR-intervals')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
end

if visual1 ==1
subplot(4,3,6)
plot(sdRR);
xlabel('time step');ylabel('RR interval (ms)')
title('Standard deviation of RR-intervals')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
end

if visual1 ==1
subplot(4,3,7)
plot(RMSSD);
xlabel('time step');
title('RMSSD')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
end

if visual1 ==1
subplot(4,3,8)
plot(NN50);
xlabel('time step');
title('NN50')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
end

if visual1 ==1
subplot(4,3,9)
plot(pNN50);
xlabel('time step');
title('pNN50')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
end

%%%%%%%%%% Frequency domain Analysis%%%%%%%%%%%%%%%%
%%%%%%%%%%%pwelch window
disp('processing frequency domain features')

for segment = 1:5
    
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.F_incre;
winlength = rr_fs*para.F_winl;
% pwelchwin = rr_fs*para.F_pwinl; %pwelch window length
% noverlap = rr_fs*para.F_over; %pwelch no overlapping sampling

%%%%%%%%%%%% instantaneous amplitude
hrv_lf = bandpass(hrv_detrend_seg{segment},[0.04 0.15],rr_fs);
hrv_hf = bandpass(hrv_detrend_seg{segment},[0.15 0.4],rr_fs);
ana_lf = hilbert(hrv_lf);
ana_hf = hilbert(hrv_hf);

while winleft + winlength< length(hrv_detrend_seg{segment})
    
    temp_iA_LF = abs(ana_lf(winleft:winleft+winlength)); % obtain the amplitude
    [~,idxmax20LF] = maxk(temp_iA_LF,floor(0.2*winlength));
    [~,idxmin20LF] = mink(temp_iA_LF,floor(0.2*winlength)); % get the indices of max and min 20%
    temp_iA_LF([idxmax20LF, idxmin20LF]) = []; % exclude the max20% and min20%
    iA_LF_{segment}(ii) = mean(temp_iA_LF);
    temp_iA_HF = abs(ana_hf(winleft:winleft+winlength)); % obtain the amplitude
    [~,idxmax20HF] = maxk(temp_iA_HF,floor(0.2*winlength));
    [~,idxmin20HF] = mink(temp_iA_HF,floor(0.2*winlength)); % get the indices of max and min 20%
    temp_iA_HF([idxmax20HF, idxmin20HF]) = []; % exclude the max20% and min20%
    iA_HF_{segment}(ii) = mean(temp_iA_HF);
%     [pxx,faxis] = pwelch(hrv_detrend(winleft:winleft+winlength),hamming(pwelchwin),noverlap,[],rr_fs);
    [pxx,faxis] = pwelch(hrv_detrend_seg{segment}(winleft:winleft+winlength),[],[],[],rr_fs);
% normalization?
%     pLFandHF = sum(pxx(find(faxis >0.04 & faxis <0.5),:),1);
    pLFandHF = 1; % no normalization 
    pLF_{segment}(ii) = sum(pxx(find(faxis >0.04 & faxis <0.15),:),1)/pLFandHF;
    pHF_{segment}(ii) = sum(pxx(find(faxis >0.15 & faxis <0.4),:),1)/pLFandHF;
    winleft = winleft+increment;
    ii = ii+1;
end
LFtoHF_{segment} = pLF_{segment}./pHF_{segment};
clearvars hrv_lf  hrv_hf ana_lf ana_hf 
end

iA_LF = cell2mat(iA_LF_);
iA_HF = cell2mat(iA_HF_);
pHF = cell2mat(pHF_);
pLF = cell2mat(pLF_);
LFtoHF = cell2mat(LFtoHF_);
% tstart = 0.5*winlength/rr_fs;
% tend = (length(pLF)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
% timeaxis = linspace(tstart,tend,length(pLF));

l1 = length(iA_LF_{1});
l2 = length(iA_LF_{2});
l3 = length(iA_LF_{3});
l4 = length(iA_LF_{4});
stagenew = [1,l1,l1+l2,l1+l2+l3,l1+l2+l3+l4];

if visual1 ==1
subplot(4,3,10)
plot(pLF);hold on
plot(pHF); hold off
title('LF and HF power, Pwelch')
legend('LF','HF')
xlabel('time step');
ylabel('power (s^2/Hz)')
vline(stagenew,'--k',{'R1','P','S','M','R2'});

subplot(4,3,11)
plot(LFtoHF);
xlabel('time step');
title('LF/HF ratio')
vline(stagenew,'--k',{'R1','P','S','M','R2'});

subplot(4,3,12)
plot(iA_LF); hold on
plot(iA_HF); hold off
xlabel('time step');
ylabel('iA (ms)');
legend('LF','HF')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
title('iA of LF and HF')
end
%save??
% saveas(gcf,['subject' num2str(subjectno) 'fig1'],'fig')

% saveas(gcf,['C:\Work\Imperial\Projects\Matlab code\figures\noedges\subject' num2str(subjectno) 'fig1'],'fig')
%%%%%%%%%%Non-Linear Analysis%%%%%%%%%%%%%%%%

%variation: just use the hrv instead of detrended version
% hrv_detrend = hrv;

% Features: Sample Entropy
disp('processing non-linear domain features')
for segment = 1:5
winleft = 1; %starting postion of the window
ii = 1;
increment = rr_fs*para.N_incre;
winlength = rr_fs*para.N_winl;


while winleft + winlength< length(hrv_detrend_seg{segment})
    hrv_window = hrv_detrend_seg{segment}(winleft:winleft+winlength);
    sd = std(hrv_window);
    SampEn1_{segment}(1,ii)= sample_entropy(hrv_window,2,0.15*sd);
    SampEn2_{segment}(1,ii)= sample_entropy(hrv_window,2,0.2*sd);
    SampEn3_{segment}(1,ii)= sample_entropy(hrv_window,1,0.2*sd);
    SampEn4_{segment}(1,ii)= sample_entropy(hrv_window,1,0.15*sd);
    MSE_{segment}(:,ii) = multiScaleEntropy(hrv_window,9);
    % alternative version of mse, 
%      = msentropy(hrv_window,2,0.15,5);
    MCSE_{segment}(:,ii) = cl_MCSE(hrv_window,2,0.07,1,10); 
%     MCSE_{segment}(:,ii) = MFE_mu(hrv_window,2,0.01,2,1,10)';
%     FuzzEnt(ii) = mmfe(hrv_window,2,1,0.15,2,5);
    MFE_{segment}(:,ii) = MFE_mu(hrv_window,2,0.01,2,1,10)';
    %poincare plot
    [PCsd1_{segment}(:,ii), PCsd2_{segment}(:,ii)]= poincare(hrv_window);
    winleft = winleft+increment;
    ii = ii+1;
end
end

% tstart = 0.5*winlength/rr_fs;
% tend = (length(SampEn1)-1)*increment/rr_fs + 0.5*winlength/rr_fs;
% SE_timeaxis = linspace(tstart,tend,length(SampEn1));

l1 = length(SampEn1_{1});
l2 = length(SampEn1_{2});
l3 = length(SampEn1_{3});
l4 = length(SampEn1_{4});
stagenew = [1,l1,l1+l2,l1+l2+l3,l1+l2+l3+l4];

SampEn1 = cell2mat(SampEn1_);
SampEn2 = cell2mat(SampEn2_);
SampEn3 = cell2mat(SampEn3_);
SampEn4 = cell2mat(SampEn4_);
% MSE_alt = cell2mat(MSE_alt_);
MSE = cell2mat(MSE_);
MFE = cell2mat(MFE_);
MCSE = cell2mat(MCSE_);
PCsd1 = cell2mat(PCsd1_);
PCsd2 = cell2mat(PCsd2_);
if visual1 ==1
figure,
subplot(3,1,1)
plot(1:numel(SampEn1),[SampEn1;SampEn2;SampEn3;SampEn4]);
xlabel('time step')
legend('m:2, r:0.15','m:2, r:0.2','m:1, r:0.2','m:1, r:0.15','Location','northeastoutside')
title(['Sample Entropy of RR, Subject ', num2str(subjectno)])
vline(stagenew,'--k',{'R1','P','S','M','R2'});

subplot(3,1,2)
plot(1:numel(SampEn1),MSE);
xlabel('time step')
title(['MSE, scale factor from 1 to 5, Subject ', num2str(subjectno)])
legend('1','2','3','4','5','Location','northeastoutside')
vline(stagenew,'--k',{'R1','P','S','M','R2'});

subplot(3,1,3)
plot(1:numel(SampEn1),MFE);
xlabel('time step')
title(['MFE, scale factor from 1 to 5, Subject ', num2str(subjectno)])
legend('1','2','3','4','5','Location','northeastoutside')
vline(stagenew,'--k',{'R1','P','S','M','R2'});
figure,
plot(1:numel(SampEn1),[PCsd1;PCsd2]);
xlabel('time step')
title(['Poincare paremeters, Subject ', num2str(subjectno)])
legend('PCsd1','PCsd2','Location','northeastoutside')
vline(stagenew,'--k',{'R1','P','S','M','R2'});

% plot the alternative version of mse
% figure,
% plot(1:numel(SampEn1),MCSE);
% xlabel('time step')
% title(['MCSE, scale factor from 1 to 5, Subject ', num2str(subjectno)])
% legend('1','2','3','4','5','Location','northeastoutside')
% vline(stagenew,'--k',{'R1','P','S','M','R2'});

end

% MSE1 = MSE(1,:);MSE2 = MSE(2,:);MSE3 = MSE(3,:);MSE4 = MSE(4,:);MSE5 = MSE(5,:);
% MFE1 = MFE(1,:);MFE2 = MFE(2,:);MFE3 = MFE(3,:);MFE4 = MFE(4,:);MFE5 = MFE(5,:);

% MSE1 = MSE(5,:);
% MFE1 = MFE(5,:);

% create categorical array
for j = 1:5
[~,edgeidx(j)] = min(abs(timeaxis - ones(1,length(timeaxis))*stage(j)));
% [~,edgeidx_hrv(j)] = min(abs(hrvaxis - ones(1,length(hrvaxis))*stage(j)));
end
stepaxis = 1:numel(SampEn1);
% Y = discretize(timeaxis,[timeaxis(edgeidx),timeaxis(end)]);
Y = discretize(stepaxis,[stagenew,stepaxis(end)]);
Y_hrv = discretize(hrvaxis,[hrvaxis(edgeidx_hrv),hrvaxis(end)]);
cate_hrv = categorical(Y_hrv,[1 2 3 4 5],{'baseline','preparation','speech','math','recovery'});
cate_feat = categorical(Y,[1 2 3 4 5],{'baseline','preparation','speech','math','recovery'});
categories(cate_feat)

% features_cell
% feat = [meanHR;sdHR;meanRR;sdRR;RMSSD;NN50;pNN50;...
%     iA_LF;iA_HF;pLF;pHF;LFtoHF;...
%     PCsd1;PCsd2;...
%     SampEn1;SampEn2;SampEn3;SampEn4;...
%     MSE1;MSE2;MSE3;MSE4;MSE5;...
%     MFE1;MFE2;MFE3;MFE4;MFE5];

feat = [meanHR;sdHR;meanRR;sdRR;RMSSD;NN50;pNN50;...
    iA_LF;iA_HF;pLF;pHF;LFtoHF;...
    PCsd1;PCsd2;...
    SampEn1;SampEn2;SampEn3;SampEn4;...
    MSE;...
    MFE;MCSE];

% feat_names = {'meanHR','sdHR','meanRR','sdRR','RMSSD','NN50','pNN50',...
%     'iA_{LF}','iA_{HF}','pLF','pHF','LFtoHF',...
%     'PCsd1','PCsd2',...
%     'SampEn1','SampEn2','SampEn3','SampEn4',...
%     'MSE','MFE'};

feat_names = {'meanHR','sdHR','meanRR','sdRR','RMSSD','NN50','pNN50',...
    'iA_{LF}','iA_{HF}','pLF','pHF','LFtoHF',...
    'PCsd1','PCsd2',...
    'SampEn1','SampEn2','SampEn3','SampEn4',...
    'MSE1','MSE2','MSE3','MSE4','MSE5','MSE6','MSE7','MSE8','MSE9',...
    'MFE1','MFE2','MFE3','MFE4','MFE5','MFE6','MFE7','MFE8','MFE9','MFE10',...
    'MCSE1','MCSE2','MCSE3','MCSE4','MCSE5','MCSE6','MCSE7','MCSE8','MCSE9','MCSE10'};

% saveas(gcf,['C:\Work\Imperial\Projects\Matlab code\figures\noedges\subject' num2str(subjectno) 'fig2'],'fig')
end
