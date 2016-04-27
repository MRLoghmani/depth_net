function img = add_flat_background( img, out_path )
%ADD_FLAT_BACKGROUND This function adds a uniform background to the image
%and saves it
%   Detailed explanation goes here
    whites = (img==255);
    bg_col = randi([min(255, max(img(not(whites)))+1),255]); % random val between max+1 and 255
    img(whites) = bg_col;
    imwrite(img, out_path);
    %imshow(img)
    %waitforbuttonpress
end

 