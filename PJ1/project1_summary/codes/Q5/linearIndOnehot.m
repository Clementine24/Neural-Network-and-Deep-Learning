function [y] = linearIndOnehot(ind,nLabels)

n = length(ind);

y = zeros(n,nLabels);

for i = 1:n
    y(i,ind(i)) = 1;
end