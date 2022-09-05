function [w] = FineTuning(w,X,y,nHidden,nLabels,lambda)
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
hiddenWeights{length(nHidden)} = w(offset+1:offset+nHidden(end)*nLabels);
hiddenWeights{end} = reshape(hiddenWeights{end},nHidden(end),nLabels);

f = lambda*sum(w.^2);
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden) + 1
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
end

% Compute Output
ip{1} = X * inputWeights;
fp{1} = ReLU(ip{1});
for h = 2:length(nHidden)
    ip{h} = fp{h-1}*hiddenWeights{h-1};
    fp{h} = ReLU(ip{h});
end
w(offset+1:offset+nHidden(end)*nLabels) = reshape(((fp{end}'*fp{end})+lambda*eye(nHidden(end)))\(fp{end}'*y),nHidden(end)*nLabels,1);