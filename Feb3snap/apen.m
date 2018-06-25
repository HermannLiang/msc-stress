function entropy = apen(pre, post, r)
% computer approximate entropy a la Steve Pincus

[N,p] = size(pre);


% how many pairs of points are closer than r in the pre space
phiM = 0;
% how many are closer in the post values
phiMplus1 = 0; 



% will be used in distance calculation
foo = zeros(N,p);

% Loop over all the points
for k=1:N
   % fill in matrix foo to contain many replications of the point in question
   for j=1:p
	foo(:,j) = pre(k,j);
   end

   % calculate the distance 
   goo = (abs( foo - pre ) <= r );

   % which ones of them are closer than r using the max norm
   if p == 1
      closerpre = goo;
   else
      closerpre = all(goo');
   end

   precount = sum(closerpre);
   phiM = phiM + log(precount);

   % of the ones that were closer in the pre space, how many are closer
   % in post also
   inds = find(closerpre);

   postcount = sum( abs( post(closerpre) - post(k) ) < r ); 
   phiMplus1 = phiMplus1 + log(postcount);
end

entropy = (phiM - phiMplus1)/N;
