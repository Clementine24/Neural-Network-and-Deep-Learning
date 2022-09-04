addpath(genpath('../'));
load digits_argumentation.mat
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
nHidden = [512 128];

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
maxIter = 15000;
stepSize = 1e-3;
lambda = 2;
Trainerror = [];
Validerror = [];
Iteration = [];
m = 0;
v = 0;
epsilon = 1e-8;
beta_1 = 0.9;
beta_2 = 0.999;
tic;
% Use ReLU activation function
funObj = @(w,i)MLPclassificationLoss_Tuning(w,X((i-1)*32+1:i*32,:),yExpanded((i-1)*32+1:i*32,:),nHidden,nLabels,lambda);
funTun = @(w)FineTuning(w,X,yExpanded,nHidden,nLabels,lambda);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        ytrain = MLPclassificationPredict_vec(w,X,nHidden,nLabels);
        Trainerror = [Trainerror sum(ytrain~=y)/n];
        Iteration = [Iteration iter];
        % Use ReLU activation function
        yhat = MLPclassificationPredict_vec(w,Xvalid,nHidden,nLabels);
        Validerror = [Validerror sum(yhat~=yvalid)/t];
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*round(n/32));
    if i == round(n/32)
        [f,g] = MLPclassificationLoss_Tuning(w,X((i-1)*32+1:end,:),yExpanded((i-1)*32+1:end,:),nHidden,nLabels,lambda);
    else
        [f,g] = funObj(w,i);
    end
    m = beta_1*m+(1-beta_1)*g;
    v = beta_2*v+(1-beta_2)*g.^2;
    m_hat = m/(1-beta_1^iter);
    v_hat = v/(1-beta_2^iter);
    w = w - stepSize*m_hat./(v_hat.^(0.5)+epsilon);
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
% Use ReLU activation function
yhat = MLPclassificationPredict_vec(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
title(['Test error = ',num2str(sum(yhat~=ytest)/t2)]);
toc