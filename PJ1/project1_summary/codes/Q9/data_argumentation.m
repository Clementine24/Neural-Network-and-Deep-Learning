addpath(genpath('../'));
load digits.mat;
% 数据增强的倍数
multiple = 3;
m = size(X,1);
% 记录新插入图像的位置
num_image = m + 1;
% 每张图像增强multiple次
for i = 1:m
    % translation
    pos = [randperm(2, 1)-1 randperm(2, 1)-1];
    img = translation(X(i,:),pos);
    X(num_image,:) = img;
    y(num_image,:) = y(i,:);
    num_image = num_image + 1;
    % rotation
    angle = rand()*30-15;
    img = rotation(X(i,:),angle);
    X(num_image,:) = img;
    y(num_image,:) = y(i,:);
    num_image = num_image + 1;
    % resize
    scale = 0.8 + 0.3*rand();
    img = resize(X(i,:),scale);
    X(num_image,:) = img;
    y(num_image,:) = y(i,:);
    num_image = num_image + 1;
end
save('digits_argumentation.mat','X','y','Xtest','ytest','Xvalid','yvalid');