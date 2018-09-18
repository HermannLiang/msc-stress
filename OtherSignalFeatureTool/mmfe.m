%% ============== Function: Multivariate multiscale fuzzy entropy ==============

function FuzzEnt = mmfe(dat,m,tau,r,order,scale_max)

% input
% dat = multivariate time series data(column vectors)
% m = embedding dimension vector, e.g.[2,2,2,...,n], n is no. of variates
% tau = yime lag vector, e.g. [1,1,1,...,n] , n is no. of variates
% r = tolerance or r value (default = 0.15)
% order = fuzzy order (default  = 2)
% scale_max = max no. of scales of sample entropy, e.g. 10 
%
% output
% FuzzEnt = fuzzy entropy value of of each scale (column vector)
%
% Example: suppose dat are 3 variates, We set m = 2 and tau = 1 for all channels
% with 10 scales: FuzzEnt = tc_mmfe(dat,[2,2,2],[1,1,1],0.15,2,10)
%
% Written by Theerasak Chanwimalueang, 17 Jan 2016

%% ============== Compute generalised mmse ==============

%***** normalise data *****
dat = zscore(dat);
r = r*sum(std(dat));

FuzzEnt = zeros(scale_max,1);
ch_max = size(dat,2);
for scale = 1:scale_max
    
    %***** coarse graining process *****  
    j_max = floor(size(dat,1)/scale); % j = grains
    y = zeros(j_max,ch_max);
    for ch = 1:ch_max
        for j = 1:j_max   
            y(j,ch) = (1/scale)*sum(dat((j-1)*scale+1:j*scale,ch));            
        end;
    end;   

    %***** construct delay vector m *****        
    m1 = m;     
    i_max = zeros(ch_max,1);
    for ch = 1:ch_max        
        i_max(ch) = size(y(:,ch),1)-tau(ch)*(m1(ch)-1);
    end;
    
    X = zeros(min(i_max),sum(m1));
    for i = 1:min(i_max)
        count = 1;
        for ch = 1:ch_max                        
            X(i,count:count+m1(ch)-1) = y(i:tau(ch):i+tau(ch)*(m1(ch)-1),ch)-mean(y(i:tau(ch):i+tau(ch)*(m1(ch)-1),ch));
            count = count+m1(ch);
        end;
    end;    
        
    %***** compute chebychev distance and accumulate events for m *****
    [n_max] = size(X,1);
    count = 0;
    for i = 1:n_max-1
        dmax = max(abs(bsxfun(@minus,X(i,:),X((i+1):n_max,:))),[],2);
        u = exp(-((dmax.^order)/r));
        count = count+sum(u);        
    end;
    d1 = count;
    d1_all = (1+(n_max-1))*(n_max-1)/2;   
    
    %***** construct delay vector m+1 *****    
    X_tmp = cell(size(dat,2),1);
    for chm=1:ch_max 
        m2 = m1;
        m2(chm) = m1(chm)+1;        
        i_max(chm) = size(y(:,chm),1)-tau(chm)*(m2(chm)-1);
        for i = 1:min(i_max)
            count = 1;
            for ch = 1:ch_max                           
                X_tmp{chm,1}(i,count:count+m2(ch)-1) = y(i:tau(ch):i+tau(ch)*(m2(ch)-1),ch)-mean(y(i:tau(ch):i+tau(ch)*(m2(ch)-1),ch));
                count = count+m2(ch);
            end;
        end;        
    end;    
    X = cat(1,X_tmp{:});   
    
    [n_max] = size(X,1);
    count = 0;
    for i = 1:n_max-1
        dmax = max(abs(bsxfun(@minus,X(i,:),X((i+1):n_max,:))),[],2);
        u = exp(-((dmax.^order)./r));
        count = count+sum(u);        
    end;
    d2 = count;
    d2_all = (1+(n_max-1))*(n_max-1)/2;    
    
    %***** compute fuzzy entropy *****   
    p1 = d1/d1_all;
    p2 = d2/d2_all;     
    
    FuzzEnt(scale,1) = -log(p2/p1);
end;

end










