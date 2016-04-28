function [ ] = convert_16to8( input_file , outpath)
%CONVERT_16TO8 Summary of this function goes here
%   Detailed explanation goes here
    co = 1/30.0;
    img = imread(input_file);
    img = double(img) * co;
    if max(img(:)) > 255
	fprintf('Max is %f\n',max(img(:)));
    end
    img(img>255)=255;
    imwrite(uint8(img), outpath);
end
