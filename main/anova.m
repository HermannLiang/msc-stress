%% ANOVA, USING ORIGINAL DATA
% pre-processing 
% clear all
clear all
load('ECGfeatures_100_47feat_16s');
mu = mean([feat{:}],2);
sig = std([feat{:}],0,2);
for i = 1:numel(feat)
    feat_std{i} = (feat{i} - mu) ./ sig;
end
% tanh normalization
for i = 1:numel(feat)
    feat_tanh{i} = 0.5.*(tanh(0.01.*(feat{i}-mu)./sig)+1);
end
[datamat,labelvec,subvec] = celldata2mat(feat,cate_feat);
%anova all
%%
for i = 1: numel(feat_names)
p_anova(i,1) = anova1(datamat(i,:),labelvec,'off');
% [~,p_ttest(i,1)] = ttest([Bmat(i,:),Amat(i,:)]);
% p_wilco(i,1) = ranksum(Bmat(i,:),Amat(i,:));
end
valid = p_anova<0.05/numel(feat_names);
 feat_names{~valid}
 
%%
% feat = feat_tanh;
barflag = 1;
cat_B = [];
cat_P= [];
cat_S = [];
cat_M = [];
cat_V = [];
% new label :{'baseline','preparation','speech','math','recovery'});
for i = 1:numel(cate_feat)
    B = setcats(cate_feat{i},{'baseline'});
    P = setcats(cate_feat{i},{'preparation'});
    S = setcats(cate_feat{i},{'speech'});
    M = setcats(cate_feat{i},{'math'});
    V = setcats(cate_feat{i},{'recovery'});
    feat_B{i} = feat{i}(:,~isundefined(B));
    cat_B = [cat_B,cate_feat{i}(:,~isundefined(B))];
    feat_P{i} = feat{i}(:,~isundefined(P));
    cat_P = [cat_P,cate_feat{i}(:,~isundefined(P))];
    feat_S{i} = feat{i}(:,~isundefined(S));
    cat_S = [cat_S,cate_feat{i}(:,~isundefined(S))];
    feat_M{i} = feat{i}(:,~isundefined(M));
    cat_M = [cat_M,cate_feat{i}(:,~isundefined(M))];
    feat_V{i} = feat{i}(:,~isundefined(V));
    cat_V = [cat_V,cate_feat{i}(:,~isundefined(V))];
end
Bmat = cell2mat(feat_B);
Pmat = cell2mat(feat_P);
Smat = cell2mat(feat_S);
Mmat = cell2mat(feat_M);
Vmat = cell2mat(feat_V);
%Baseline vs Preparation
for i = 1: numel(feat_names)
p_anova(i,1) = anova1([Bmat(i,:),Pmat(i,:)],[ones(1,size(Bmat,2)),2*ones(1,size(Pmat,2))],'off');
% [~,p_ttest(i,1)] = ttest([Bmat(i,:),Pmat(i,:)]);
p_wilco(i,1) = ranksum(Bmat(i,:),Pmat(i,:));
end
%Baseline vs Speech
for i = 1: numel(feat_names)
p_anova(i,2) = anova1([Bmat(i,:),Smat(i,:)],[ones(1,size(Bmat,2)),2*ones(1,size(Smat,2))],'off');
p_wilco(i,2)= ranksum(Bmat(i,:),Smat(i,:));
end
%Baseline vs Math
for i = 1: numel(feat_names)
p_anova(i,3) = anova1([Bmat(i,:),Mmat(i,:)],[ones(1,size(Bmat,2)),2*ones(1,size(Mmat,2))],'off');
p_wilco(i,3)= ranksum(Bmat(i,:),Mmat(i,:));
end
%Baseline vs Speech+math
for i = 1: numel(feat_names)
p_anova(i,4) = anova1([Bmat(i,:),Smat(i,:),Mmat(i,:)],[ones(1,size(Bmat,2)),2*ones(1,size(Smat,2)+size(Mmat,2))],'off');
p_wilco(i,4)= ranksum(Bmat(i,:),[Smat(i,:),Mmat(i,:)]);
end
%Preparation vs speech
for i = 1: numel(feat_names)
p_anova(i,5) = anova1([Pmat(i,:),Smat(i,:)],[ones(1,size(Pmat,2)),2*ones(1,size(Smat,2))],'off');
p_wilco(i,5)= ranksum(Pmat(i,:),Smat(i,:));
end
%Preparation vs math
for i = 1: numel(feat_names)
p_anova(i,6) = anova1([Pmat(i,:),Mmat(i,:)],[ones(1,size(Pmat,2)),2*ones(1,size(Mmat,2))],'off');
p_wilco(i,6)= ranksum(Pmat(i,:),Mmat(i,:));
end
%Preparation vs speech+math
for i = 1: numel(feat_names)
p_anova(i,7) = anova1([Pmat(i,:),Smat(i,:),Mmat(i,:)],[ones(1,size(Pmat,2)),2*ones(1,size(Mmat,2)+size(Smat,2))],'off');
p_wilco(i,7)= ranksum(Pmat(i,:),[Smat(i,:),Mmat(i,:)]);
end
%speech vs math
for i = 1: numel(feat_names)
p_anova(i,8) = anova1([Smat(i,:),Mmat(i,:)],[ones(1,size(Smat,2)),2*ones(1,size(Mmat,2))],'off');
p_wilco(i,8)= ranksum(Smat(i,:),Mmat(i,:));
end
%baseline vs recovery
for i = 1: numel(feat_names)
p_anova(i,9) = anova1([Bmat(i,:),Vmat(i,:)],[ones(1,size(Bmat,2)),2*ones(1,size(Vmat,2))],'off');
p_wilco(i,9)= ranksum(Bmat(i,:),Vmat(i,:));
end

valid = p_anova<0.05/numel(feat_names); % reject null hypothesis at the bonferroni corrected p value
% valid = p_anova<0.05; % reject null hypothesis at the bonferroni corrected p value
valid_wil = p_wilco<0.05/numel(feat_names); % reject null hypothesis at the bonferroni corrected p value
score_anova = sum(double(valid(:,1:8)),2);
score_wil = sum(double(valid_wil(:,1:8)),2);
save('score_anova.mat','score_anova');
save('score_wil.mat','score_wil');

%%
%BPS
counts = sum(double(valid(:,[1 2 5])),2);
abd_feat(:,1) = counts<=0;
%BPM
counts = sum(double(valid(:,[1 3 6])),2);
abd_feat(:,2) = counts<=0;
%BPS/M
counts = sum(double(valid(:,[4 7])),2);
abd_feat(:,3) = counts<=0;
%BP
counts = sum(double(valid(:,[1])),2);
abd_feat(:,4) = counts<=0;
%BS/M
counts = sum(double(valid(:,[4])),2);
abd_feat(:,5) = counts<=0;
%PS/M
counts = sum(double(valid(:,[7])),2);
abd_feat(:,6) = counts<=0;

counts = sum(double(valid()),2); % count n.o features that alt hypo holds
counts_wil = sum(double(valid_wil),2);
% abd_feat = counts<=0; % indices of features should be removed
% abd_feat = counts==0; % indices of features should be removed
if barflag ==1 
figure,
c = categorical(feat_names(1:7));
bar(c,p_anova(1:7,:));
title('ANOVA: Time domain features')
legend('B vs P','B vs S','B vs M','B vs S+M','P vs S','P vs M','P vs S+M', 'S vs M','B vs V','Location','northeastoutside');

figure,
c = categorical(feat_names(8:12));
bar(c,p_anova(8:12,:));
title('ANOVA: Frequency domain features')
legend('B vs P','B vs S','B vs M','B vs S+M','P vs S','P vs M','P vs S+M', 'S vs M','B vs V','Location','northeastoutside');

figure,
c = categorical(feat_names(13:20));
bar(c,p_anova(13:20,:));
title('ANOVA: Non-linear features')
legend('B vs P','B vs S','B vs M','B vs S+M','P vs S','P vs M','P vs S+M', 'S vs M','B vs V','Location','northeastoutside');

% wilco test
figure,
c = categorical(feat_names(1:7));
bar(c,p_wilco(1:7,:));
title('Wilcoxon: Time domain features')
legend('B vs P','B vs S','B vs M','B vs S+M','P vs S','P vs M','P vs S+M', 'S vs M','B vs V','Location','northeastoutside');

figure,
c = categorical(feat_names(8:12));
bar(c,p_wilco(8:12,:));
title('Wilcoxon: Frequency domain features')
legend('B vs P','B vs S','B vs M','B vs S+M','P vs S','P vs M','P vs S+M', 'S vs M','B vs V','Location','northeastoutside');

figure,
c = categorical(feat_names(13:20));
bar(c,p_wilco(13:20,:));
title('Wilcoxon: Non-linear features')
legend('B vs P','B vs S','B vs M','B vs S+M','P vs S','P vs M','P vs S+M', 'S vs M','B vs V','Location','northeastoutside');

end


save('abd_feat.mat','abd_feat');
% xlswrite('p_anova.xlsx',p_anova);
% xlswrite('p_wilco.xlsx',p_wilco);

% 
% for idx = 1:find(abd_feat)
%  feat_names{abd_feat}
% end

