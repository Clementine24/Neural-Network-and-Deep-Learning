% softmax function
function y = softmax(x)
x_exp = exp(x);
y = x_exp./repmat(sum(x_exp, 2),1,size(x,2)); % 注意这里要求X传入时类别是按列排序的
end