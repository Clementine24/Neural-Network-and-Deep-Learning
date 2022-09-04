function [img] = translation(data, pos)
% 先转换为图像格式
origin_img = reshape(data, 16, 16);
% 对图像平移
img = imtranslate(origin_img, pos, 'method', 'bilinear');
% 将数据展开回原来的格式
img = reshape(img, 1, 256);
end

