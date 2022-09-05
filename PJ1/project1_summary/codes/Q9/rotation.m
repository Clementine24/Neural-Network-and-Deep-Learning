function [img] = rotation(data, angle)
% 先转换为图像格式
origin_img = reshape(data, 16, 16);
% 旋转图像
img = imrotate(origin_img, angle, 'bilinear');
% 将图像恢复为要求格式
img = imresize(img,16/size(img,1));
% 将数据展开回原来的格式
img = reshape(img, 1, 256);
end
