%% ============== Function: Multivariate multiscale sample entropy ==============

function se = mmse(dat,m,tau,r,scale_max)

% input
% dat = original data(each channel of data is in each column vector)
% m =  embedding vector, e.g.[2,2,2,...,n], n is no. of variates
% tau = time lag vector, e.g. [1,1,1,...,n] , n is no. of variates
% r = tolerance, the ratio of the standard deviation (e.g. 0.10-0.20; default = 0.15)
% scale_max = max no. of scales of sample entropy, e.g. 10  
%
% output
% se = mmse data of each scale
%
% Example: suppose dat = signals of 3 channels. We set m = 2 and tau = 1 for all channels
% with 10 scales: se = tc_mmse(dat,[2,2,2],[1,1,1],0.15,10)
% Written by Theerasak Chanwimalueang, 14 July 2014, mooicy@gmail.com

%% ============== Compute mmse ==============

%***** normalise data *****
% dat = zscore(dat);
dat = fnormc(dat);
r = r*sum(std(dat));

se = zeros(scale_max,1);
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
            X(i,count:count+m1(ch)-1) = y(i:tau(ch):i+tau(ch)*(m1(ch)-1),ch);
            count = count+m1(ch);
        end;
    end;    
    
    %***** compute chebysheb distance of vector m *****
    [n_max] = size(X,1);
    count = 0;
    for i = 1:n_max-1
        dmax = max(abs(bsxfun(@minus,X(i,:),X((i+1):n_max,:))),[],2);
        count = count+length(dmax(dmax<=r));
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
                X_tmp{chm,1}(i,count:count+m2(ch)-1) = y(i:tau(ch):i+tau(ch)*(m2(ch)-1),ch);
                count = count+m2(ch);
            end;
        end;        
    end;    
    X = cat(1,X_tmp{:});    
    
    %***** compute chebysheb distance of vector m+1 *****
    [n_max] = size(X,1);
    count = 0;
    for i = 1:n_max-1
        dmax = max(abs(bsxfun(@minus,X(i,:),X((i+1):n_max,:))),[],2);
        count = count+length(dmax(dmax<=r));
    end;
    d2 = count;
    d2_all = (1+(n_max-1))*(n_max-1)/2;   
    
    %***** compute sample entropy *****   
    p1 = d1/d1_all;
    p2 = d2/d2_all;     
    
    se(scale,1) = -log(p2/p1);
end;

end










