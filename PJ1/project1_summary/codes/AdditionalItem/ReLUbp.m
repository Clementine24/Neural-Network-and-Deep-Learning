function g = ReLUbp(x)
g = zeros(size(x));
g(x>0) = 1;
end