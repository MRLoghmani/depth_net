function [ b_subs ] = addBackground(object ,backgroundImg, show )
b=imread(backgroundImg);
b=imresize(b,[256 256]);
maxB=max(max(b));
b_uint8=uint8(255*double(b)./(double(maxB)));
%object=imread(obImg);
objectStripped=object;
objectStripped(objectStripped==255) = 0;
maxOb=max(max(objectStripped));
%%% I scale and shift the background image with the following equation:
% X = Y*M + K
% M = (Y - 255) / (X - 255)
% K = 255 - 255*M
%That will shift a point to another point but keeping maximum=255
%So, let me shift the background center value to the object max value:
[w,h] = size(b);
mean=mean2(b_uint8(w/2-5:w/2+5, h/2-5:h/2+5));
%Here Y=mean, X=maxOb_stripped
M = (double(maxOb) - 255.0) / (mean  - 255.0);
K = 255 - 255 * M;
b_shifted= uint8(double(b_uint8).*M + K);

b_subs=b_shifted;
%Substituting all pixels of object that are not zero!
for i = 1:numel(objectStripped)
    if objectStripped(i) ~= 0
      b_subs(i) = objectStripped(i);  
    end
end
if nargin == 3
figure;
imshow(b_subs);
end 
%figure;
%imshow(b_uint8);
%figure;
%imshow(object);
%figure;
%imshow(b_shifted);
%figure;
%imshow(b_subs);
end

