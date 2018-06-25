function [ feats_est ] = tanhEst( feats )
% feats: M observations x N dimensions

meanMat = ones(size(feats,1),1)*mean(feats,1);
stdMat = ones(size(feats,1),1)*std(feats,1);
feats_est = 0.5.*(tanh(0.01.*(feats-meanMat)./stdMat)+1);
end

