function [y] = SoftmaxPredict(w,X,nHidden,nLabels)
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1)); %因为输入的X已经是加过bias项的了，因此不需要再处理
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),(nHidden(h-1)+1),nHidden(h));
  offset = offset+(nHidden(h-1)+1)*nHidden(h);
end
hiddenWeights{length(nHidden)} = w(offset+1:offset+(nHidden(end)+1)*nLabels);
hiddenWeights{end} = reshape(hiddenWeights{end},(nHidden(end)+1),nLabels);

% Compute Output
ip{1} = X * inputWeights;
fp{1} = [ones(nInstances,1) ReLU(ip{1})];
for h = 2:length(nHidden)
    ip{h} = fp{h-1}*hiddenWeights{h-1};
    fp{h} = [ones(nInstances,1) ReLU(ip{h})];
end
y = softmax(fp{end}*hiddenWeights{end});

[v,y] = max(y,[],2);  %这里返回的是每一个预测结果，即预测概率最大的
%y = binary2LinearInd(y);
