function [f,g] = CNN_Loss(w,X,y,nHidden,nLabels,kernel_size,padding)

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

% f = 0;
if nargout > 1
    gKernel = zeros(kernel_size, kernel_size);
    gInput = zeros(size(inputWeights));
    for h = 2:length(nHidden) + 1
       gHidden{h-1} = zeros(size(hiddenWeights{h-1})); 
    end
end

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
    yhat = fp{end}*hiddenWeights{end};

    relativeErr = yhat - y;
    f = sum(sum(relativeErr.^2,1),2);

    if nargout > 1
        err = 2 * relativeErr;
        for h = length(nHidden):-1:1
           gHidden{h} = gHidden{h} + fp{h}' * err;
           err = ReLUbp(fp{h}) .* (err * hiddenWeights{h}'); %一定要加上括号先计算右边部分
        end
        gInput = gInput + x' * err;
        % 计算卷积核梯度
        err = err * inputWeights';
        err = reshape(err, size(conv_out));
        reverse_x = reshape(X(i,end:-1:1),16,16);
        gKernel = gKernel + conv2(reverse_x, err, 'valid');
    end
end


% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:kernel_size^2) = gKernel(:);
    offset = kernel_size^2;
    g(offset + 1:offset + (16-kernel_size+2*padding+1)^2*nHidden(1)) = gInput(:);
    offset = offset + (16-kernel_size+2*padding+1)^2*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
        offset = offset+nHidden(h-1)*nHidden(h);
    end
    g(offset+1:offset+nHidden(end)*nLabels) = gHidden{end};
end

