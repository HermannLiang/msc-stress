function [ output_args ] = fPlotConfusion( predict_label,test_label )
% fPlotConfusion
% Generate Confusion Matrix.
label_class = unique(test_label);
target = zeros(length(label_class),length(test_label));
predict = zeros(length(label_class),length(test_label));

% Set the target and predict matrix
for  i = 1:label_class(end)
    target(i,test_label==i)= 1;
    predict(i,predict_label==i)= 1;
end
figure
plotconfusion(target,predict);
end

