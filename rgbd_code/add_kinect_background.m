function add_kinect_background( img, bg_img, bg_mean, out_path )
%ADD_FLAT_BACKGROUND This function adds a uniform background to the image
%and saves it
%   Detailed explanation goes here
    obj = (img ~= 255);
    maxob = double(max(img(obj)));
    M = (maxob - 255.0) / (bg_mean  - 255.0);
    if isempty(M)
        return
    end
    K =  255 - 255 * M;
    b_shifted = uint8(double(bg_img).*M + K);
    img = b_shifted .* uint8(not(obj)) + img .* uint8(obj);
    imwrite(img, strrep(out_path, '.png', '_kinect.png'));
    %imshow(img)
    %waitforbuttonpress
end

 
