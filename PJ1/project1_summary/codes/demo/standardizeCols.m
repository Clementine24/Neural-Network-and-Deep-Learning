function [S, mu, sigma2] = standardizeCols(M, mu, sigma2)
% function [S,mu,sigma2] = standardize(M, mu, sigma2)
% Make each column of M be zero mean, std 1.
%
% If mu, sigma2 are omitted, they are computed from M

[nrows ncols] = size(M);

M = double(M);
if nargin < 2 %只传入一个参数
  mu = mean(M);
  sigma2 = std(M);
  ndx = find(sigma2 < eps);  %eps=2.2204e-16，猜测这里是为了处理那些全为0或只有一个不为0的行
  sigma2(ndx) = 1;
end

S = M - repmat(mu, [nrows 1]);
if ncols > 0
S = S ./ repmat(sigma2, [nrows 1]);
end
