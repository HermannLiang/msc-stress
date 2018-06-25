function nlabels = killisol(labels, nconseq, goodlabel)
% nlabels = killisol(labels, nconseq, goodlabel)
% find any islands of not <goodlabel> labels that is
% smaller or equal to nconseq

% pad the data with a zero. Later, this will kill off trailing ones
labels = [labels(:);0];
nlabels = labels;

% find the bad labels
zs = find(labels ~= goodlabel);
% find the indices of these in our original data
goo = 1:length(labels);
goo = goo(zs);

% find the lengths of segments between bad labels
lens=diff( [0;goo(:)]) -1;
% find the short segments
foo = find(lens <= nconseq & lens > 0);
% changes isolated labels to badlabels


for k=1:length(foo)
  koo = goo(foo(k));
  for j=1:(lens(foo(k)))
    nlabels(koo-j) = 0;
  end
end
    
nlabels = nlabels(1:(length(nlabels)-1));



