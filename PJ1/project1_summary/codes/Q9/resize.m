function [img] = resize(data, scale)
% 先转换为图像格式
origin_img = reshape(data, 16, 16);
% 分情况讨论resize
img = imresize(origin_img, scale, 'bilinear');
[m, n] = size(img);
if scale < 1
    % 对图像缺失的维度进行pad
    img = padarray(img, [floor((16-n)/2) floor((16-m)/2)],0, 'post');
    img = padarray(img, [16-size(img,2) 16-size(img,1)],0, 'pre');
elseif scale > 1
    % 裁剪多余的维度
    img = imcrop(img, [floor(n/2)-7, floor(m/2)-7 15 15]);
% 将数据展开回原来的格式
end
% scale
% size(img)
img = reshape(img, 1, 256);
end
