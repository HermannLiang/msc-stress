function [ output ] = fnormc( input )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    M = max(input,[],1);
    m = min(input,[],1);
    u = mean(input,1);
    
    M = ones(size(input,1),1)*M;
    m = ones(size(input,1),1)*m;
    u = ones(size(input,1),1)*u;
    
    output = (input-m)./(M-m);

end

