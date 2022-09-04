load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);  %将标签变为one-hot矩阵，但是0变为-1
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X); %按列进行标准化，即对每一个特征进行标准化
X = [ones(n,1) X];  %加bias项
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);  %注意这里不能向前面一样只传入一个参数，
Xvalid = [ones(t,1) Xvalid];  %因为必须和前面训练数据使用相同的变换，因此使用前面返回的参数
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
% Choose one hidden layer but different units number
% nHidden = [10];
% nHidden = [16];
% nHidden = [32];
nHidden = [64];
% nHidden = [128];
% nHidden = [256];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)  %这里很巧的是MATLAB的倒序只会生成空序列
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;   %请用end代替-1，可以使用end进行数值运算
w = randn(nParams,1);
w(1:d*nHidden(1)) = w(1:d*nHidden(1))*sqrt(2./(d*nHidden(1)));
offset = d*nHidden(1);
for h = 2:length(nHidden)
    w(offset+1:offset+nHidden(h-1)*nHidden(h)) = w(offset+1:offset+ ...
        nHidden(h-1)*nHidden(h))* sqrt(2./(nHidden(h-1)*nHidden(h)));
    offset = offset + nHidden(h-1)*nHidden(h);
end
w(offset+1:nParams) = w(offset+1:nParams)*sqrt(2./(nHidden(end)*nLabels));

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
tic;
% Use ReLU activation function
funObj = @(w,i)MLPclassificationLoss_ReLU(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        % Use ReLU activation function
        yhat = MLPclassificationPredict_ReLU(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
%         e=sprintf('%d ', w);
%         fprintf('Weight: %s\n', e);
    end
    
    i = ceil(rand*n); %从n个样本里随机抽取一个
    [f,g] = funObj(w,i); 
    w = w - stepSize*g;
end

% Evaluate test error
% Use ReLU activation function
yhat = MLPclassificationPredict_ReLU(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
toc