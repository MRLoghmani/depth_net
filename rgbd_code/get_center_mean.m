function [ m_val ] = get_center_mean( img, k )
%GET_CENTER_MEAN Computes mean of image center
%   Area is 2k+1 x 2k+1
    [h,w] = size(img);
    h = uint8(h/2);
    w = uint8(w/2);
    center = img(h-k:h+k, w-k:w+k);
    m_val = mean(center(:));
end

