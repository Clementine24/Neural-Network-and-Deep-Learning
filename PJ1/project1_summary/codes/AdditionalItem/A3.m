addpath(genpath('../'));
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
% nHidden = [512];

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
lambda = 0.1;
Trainerror = [];
Validerror = [];
Iteration = [];
MinValidError = 1.0;
alpha = 80;
tic;

% 使用Mini-Batch Gradient Descent
funObj = @(w,i)MLPclassificationLoss_Tuning(w,X((i-1)*16+1:i*16,:),yExpanded((i-1)*16+1:i*16,:),nHidden,nLabels,lambda);
funTun = @(w)FineTuning(w,X,yExpanded,nHidden,nLabels,lambda);
for iter = 1:maxIter
    if mod(iter-1,100) == 0  % 每100轮检测一次验证集错误率
        yhat = MLPclassificationPredict_vec(w,Xvalid,nHidden,nLabels);
        ValidError = sum(yhat~=yvalid)/t;
        if ValidError < MinValidError
            MinValidError = ValidError;
        elseif 100*(ValidError/MinValidError-1) > alpha
            fprintf('It is time to stop train early, train iteration = %d, and the validation error is %f\n',iter-1,ValidError);
            break;
        end
    end
    if mod(iter-1,round(maxIter/20)) == 0
        ytrain = MLPclassificationPredict_vec(w,X,nHidden,nLabels);
        Trainerror = [Trainerror sum(ytrain~=y)/n];
        Iteration = [Iteration iter];
        Validerror = [Validerror ValidError];
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,ValidError);
    end
    
%     i = ceil(rand*n); %从n个样本里随机抽取一个
    i = ceil(rand*floor(n/16));
    [f,g] = funObj(w,i); 
    w = w - stepSize*g;
    if mod(iter,500) == 0
        w = funTun(w);
    end
end
figure()
plot(Iteration,Trainerror,'g',Iteration,Validerror,'m')
legend('train set error','valid set error');
xlabel(['Train iteration, step size=',num2str(stepSize)]);
ylabel('Error');

% Evaluate test error
yhat = MLPclassificationPredict_vec(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
title(['Test error = ',num2str(sum(yhat~=ytest)/t2)]);
toc