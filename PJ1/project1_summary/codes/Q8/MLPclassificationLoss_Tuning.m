function [f,g] = MLPclassificationLoss_Tuning(w,X,y,nHidden,nLabels,lambda)

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
yhat = fp{end}*hiddenWeights{end};

relativeErr = yhat - y;
f = f + sum(sum(relativeErr.^2,1),2);

if nargout > 1
    err = 2 * relativeErr;
    for h = length(nHidden):-1:1
       gHidden{h} = fp{h}' * err + 2*lambda*hiddenWeights{h};
       err = ReLUbp(fp{h}) .* (err * hiddenWeights{h}'); %一定要加上括号先计算右边部分
    end
    gInput = X' * err;
end


% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gHidden{end};
end

