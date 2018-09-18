function [datamat,labelvec,subvec] = celldata2mat(datacell,labelcell)
%Transform data from cell to matrix
% in most case, cell2mat did the job but unfortunately, it didn't works
% with label though
% datamat: variable*dimension, your data matrix
% labelmat: one dimension label vector 
% subjectvec long subject label vector;
datamat = cell2mat(datacell);
%   Detailed explanation goes here
labelvec = [];
subvec= [];
for i = 1:numel(labelcell)
    labelvec = [labelvec,labelcell{i}];
    subvec = [subvec,i*ones(1,length(labelcell{i}))]; 
end
end

