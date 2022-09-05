function [y] = CNN_Predict(w,X,nHidden,nLabels,kernel_size,padding)
[nInstances,~] = size(X);

% Form Weights
kernel = reshape(w(1:kernel_size^2),kernel_size,kernel_size);
offset = kernel_size^2;
inputWeights = reshape(w(offset + 1:offset + (16-kernel_size+2*padding+1)^2*nHidden(1)),(16-kernel_size+2*padding+1)^2,nHidden(1));
offset = offset + (16-kernel_size+2*padding+1)^2*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
hiddenWeights{length(nHidden)} = w(offset+1:offset+nHidden(end)*nLabels);
hiddenWeights{end} = reshape(hiddenWeights{end},nHidden(end),nLabels);


% Compute Output
for i = 1:nInstances
    x = reshape(X(i,:),16,16);
    pad_x = padarray(x, [padding, padding], 0, 'both');
    conv_out = conv2(pad_x,kernel,'valid');
    x = reshape(conv_out,1,size(conv_out,1)^2);
    ip{1} = x * inputWeights;
    fp{1} = ReLU(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = ReLU(ip{h});
    end
    y(i,:) = fp{end}*hiddenWeights{end};
end

[v,y] = max(y,[],2);  %这里返回的是每一个预测结果，即预测概率最大的
%y = binary2LinearInd(y);
