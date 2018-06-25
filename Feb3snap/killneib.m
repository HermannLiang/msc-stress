function nlabels = killneib( labels, pre, post, goodlabel )
% nlabels = killneibs( labels, pre, post, goodlabel )
% labels as invalid any points that are within <pre> previous
% or <post> following invalid points in <labels>

nlabels = ones(size(labels));
binds = find( labels ~= goodlabel);
nlabels(binds) = 0;

for k=1:post
  nbinds = min(length(labels), binds + k);
  nlabels(nbinds) = 0;
end

for k=1:pre
  nbinds = max(1, binds - k);
  nlabels(nbinds) = 0;
end

