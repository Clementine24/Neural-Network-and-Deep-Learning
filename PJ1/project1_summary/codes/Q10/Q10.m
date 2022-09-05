addpath(genpath('../'));
load digits.mat;
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels); 
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X); %按列进行标准化，即对每一个特征进行标准化
% X = [ones(n,1) X];  %加bias项
% d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma); 
% Xvalid = [ones(t,1) Xvalid]; 
Xtest = standardizeCols(Xtest,mu,sigma);
% Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [64];
kernel_size = 7;
stride = 1;
padding = 0;

% Count number of parameters and initialize weights 'w'
nParams = kernel_size ^ 2;  % kernel params
nParams = nParams + nHidden(1)*((16-kernel_size+2*padding)/stride+1)^2; % 第一个全连接层
for h = 2:length(nHidden) 
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;   %请用end代替-1，可以使用end进行数值运算
w = randn(nParams,1)*0.001;

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
Trainerror = [];
Validerror = [];
Iteration = [];
funObj = @(w,i)CNN_Loss(w,X(i,:),yExpanded(i,:),nHidden,nLabels,kernel_size,padding);  %因为采用的是随机梯度下降，因此每次只使用一个样本去更新
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        ytrain = CNN_Predict(w,X,nHidden,nLabels,kernel_size,padding);
        Trainerror = [Trainerror sum(ytrain~=y)/n];
        Iteration = [Iteration iter];
        % Use ReLU activation function
        yhat = CNN_Predict(w,Xvalid,nHidden,nLabels,kernel_size,padding);
        Validerror = [Validerror sum(yhat~=yvalid)/t];
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n); %从n个样本里随机抽取一个
    [f,g] = funObj(w,i); 
    w = w - stepSize*g;
end
figure()
plot(Iteration,Trainerror,'g',Iteration,Validerror,'m')
legend('train set error','valid set error');
xlabel(['Train iteration, step size=',num2str(stepSize)]);
ylabel('Error');


% Evaluate test error
yhat = CNN_Predict(w,Xtest,nHidden,nLabels,kernel_size, padding);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
title(['Test error = ',num2str(sum(yhat~=ytest)/t2)]);