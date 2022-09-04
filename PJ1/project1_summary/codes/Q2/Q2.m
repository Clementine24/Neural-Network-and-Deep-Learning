load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearIndOnehot(y,nLabels);  %将标签变为one-hot矩阵
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
% nHidden = [10];
% nHidden = [64];
nHidden = [256];
% nHidden = [512];


% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)  %这里很巧的是MATLAB的倒序只会生成空序列
    nParams = nParams+(nHidden(h-1)+1)*nHidden(h); % 加1是为了引入bias项,后面同理
end
nParams = nParams+(nHidden(end)+1)*nLabels;   %请用end代替-1，可以使用end进行数值运算
% 对初始化权重进行方差缩放
w = randn(nParams,1);
w(1:d*nHidden(1)) = w(1:d*nHidden(1))*sqrt(2./(d*nHidden(1)));
offset = d*nHidden(1);
for h = 2:length(nHidden)
    w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = w(offset+1:offset+ ...
        (nHidden(h-1)+1)*nHidden(h))* sqrt(2./((nHidden(h-1)+1)*nHidden(h)));
    offset = offset + (nHidden(h-1)+1)*nHidden(h);
end
w(offset+1:nParams) = w(offset+1:nParams)*sqrt(2./((nHidden(end)+1)*nLabels));

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
beta = 0.9;
% decay = 0.999;
% 记录训练过程
Trainerror = [];
Validerror = [];
Iteration = [];
w_pred = w;
tic;
% Use Softmax classification
funObj = @(w,i)SoftmaxLoss(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        % Use Softmax classification
        yhat = SoftmaxPredict(w,Xvalid,nHidden,nLabels);
        ytrain = SoftmaxPredict(w,X,nHidden,nLabels);
        Trainerror = [Trainerror sum(ytrain~=y)/n];
        Validerror = [Validerror sum(yhat~=yvalid)/t];
        Iteration = [Iteration iter];
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
%         stepSize = stepSize * decay;
    end
    
    i = ceil(rand*n); %从n个样本里随机抽取一个
    [f,g] = funObj(w,i); 
    w_now = w;
    w = w - stepSize*g + beta*(w_now-w_pred);
    w_pred = w_now;
end
plot(Iteration,Trainerror,'g',Iteration,Validerror,'m')
legend('train set error','valid set error');
xlabel('Train iteration(step size=1e-4)');
ylabel('Error');

% Evaluate test error
% Use Softmax classification
yhat = SoftmaxPredict(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);
toc