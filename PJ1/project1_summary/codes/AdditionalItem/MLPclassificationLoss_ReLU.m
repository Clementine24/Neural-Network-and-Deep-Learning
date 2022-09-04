function [f,g] = MLPclassificationLoss_ReLU(w,X,y,nHidden,nLabels)

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

f = 0;
if nargout > 1
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden)
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
    gOutput = zeros(size(outputWeights));
end

% Compute Output
for i = 1:nInstances   %这里使用的是一次只关注一个样本，因此后面每次W的梯度都要自加和，本来就是梯度相当于每个样本的梯度加上去
    ip{1} = X(i,:)*inputWeights;
    fp{1} = ReLU(ip{1});  
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = ReLU(ip{h});
    end
    yhat = fp{end}*outputWeights;
    
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);  %f计算的是平方差损失，但是这里计算的是累计Loss，应该求平均才是
    
    if nargout > 1
        err = 2*relativeErr;

        % Output Weights
        for c = 1:nLabels
            gOutput(:,c) = gOutput(:,c) + err(c)*fp{end}'; %这里如果是矩阵的话应该是fp在前面，因为这里err是一个数不碍事，后面注意（因为这里的样本是按行排列）
        end

        if length(nHidden) > 1
            % Last Layer of Hidden Weights
            clear backprop  % 每次要清干净之前的
            for c = 1:nLabels
                %这里用了一个很蠢的做法，本来是err转置点乘W的，但是变成了循环每次只乘一个err数，这也导致了backprop出现了nlabels个行维度，本来应该是1的
                backprop(c,:) = err(c)*(ReLUbp(ip{end}).*outputWeights(:,c)'); %计算Z_L的梯度，注意这里是点积
                gHidden{end} = gHidden{end} + fp{end-1}'*backprop(c,:); %因为这里一次只使用了一条数据，第一个维度为1，为了匹配这个维度
            end
            backprop = sum(backprop,1); %现在的backprop维度又正常了，为1*noutputs

            % Other Hidden Layers
            for h = length(nHidden)-2:-1:1
                backprop = (backprop*hiddenWeights{h+1}').*ReLUbp(ip{h+1}); %注意这里W本来就比IP快一个单位，也是点乘
                gHidden{h} = gHidden{h} + fp{h}'*backprop;
            end

            % Input Weights
            backprop = (backprop*hiddenWeights{1}').*ReLUbp(ip{1});
            gInput = gInput + X(i,:)'*backprop;
        else
           % Input Weights
            for c = 1:nLabels
                gInput = gInput + err(c)*X(i,:)'*(ReLUbp(ip{end}).*outputWeights(:,c)');
            end
        end

    end
    
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
    g(offset+1:offset+nHidden(end)*nLabels) = gOutput(:);
end

