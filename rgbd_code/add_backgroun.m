object = imread('/home/enoon/DepthNet/scripts/rgbd_code/desk_instance_01_14_0001.png');
input_folder = '/home/enoon/Downloads/toSelect/8BitScenes/background/selected/*.png';
out_folder = '/home/enoon/Downloads/toSelect/8BitScenes/background/selected/rendered'
bgs = dir(input_folder);
k=0
for file = bgs'
    joined = addBackground(object, strcat('/home/enoon/Downloads/toSelect/8BitScenes/background/selected/',file.name));
    imwrite(joined, strcat('image_',num2str(k),'.png'));
    k = k+1
end