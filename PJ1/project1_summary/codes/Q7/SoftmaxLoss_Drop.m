function [f,g] = SoftmaxLoss_Drop(w,X,y,nHidden,nLabels,drop_prob)

[nInstances,nVars] = size(X);
% fprintf('X shape is %f,%f\n',size(X,1),size(X,2));

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1)); %因为输入的X已经是加过bias项的了，因此不需要再处理
offset = nVars*nHidden(1);
% 生成输入权重矩阵的drop-out mask矩阵
mask{1} = [ones(nInstances,1) repmat(rand(1,nVars-1) > drop_prob(1),nInstances,1)]/(1-drop_prob(1));
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),(nHidden(h-1)+1),nHidden(h));
  offset = offset+(nHidden(h-1)+1)*nHidden(h);
  mask{h} = repmat(rand(1,nHidden(h-1)) > drop_prob(h),nInstances,1)/(1-drop_prob(h));
end
hiddenWeights{length(nHidden)} = w(offset+1:offset+(nHidden(end)+1)*nLabels);
hiddenWeights{end} = reshape(hiddenWeights{end},(nHidden(end)+1),nLabels);
mask{length(nHidden)+1} = repmat(rand(1,nHidden(end)) > drop_prob(length(nHidden)+1),nInstances,1)/(1-drop_prob(length(nHidden)+1));

if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden) + 1
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
end

% Compute Output
ip{1} = X * inputWeights;
fp{1} = [ones(nInstances,1) mask{2}.*ReLU(ip{1})]; % 做drop-out处理
for h = 2:length(nHidden)
    ip{h} = fp{h-1}*hiddenWeights{h-1};
    fp{h} = [ones(nInstances,1) mask{h+1}.*ReLU(ip{h})];
end
yhat = softmax(fp{end}*hiddenWeights{end}); % 加入softmax分类

f = sum(-log(yhat(y==1)));  % 计算交叉熵损失，为了减小计算量，这里是挑出来再取对数

if nargout > 1
    err = yhat - y;
    for h = length(nHidden):-1:1
       gHidden{h} = fp{h}' * err;
       % 反向传播要记得去掉丢弃的点
       err = mask{h+1}.*ReLUbp(fp{h}(:,2:nHidden(h)+1)) .* (err * (hiddenWeights{h}(2:nHidden(h)+1,:))')*(1-drop_prob(h+1)); %一定要加上括号先计算右边部分
       % fprintf('error %d shape is %f,%f\n',h,size(err,1),size(err,2));
    end
    gInput = X' * err;
    % fprintf('ginput shape is %f,%f\n',size(gInput,1),size(gInput,2));
end


% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = gHidden{h-1};
        offset = offset+(nHidden(h-1)+1)*nHidden(h);
    end
    g(offset+1:offset+(nHidden(end)+1)*nLabels) = gHidden{end};
end

