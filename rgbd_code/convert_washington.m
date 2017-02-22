filelist = importdata('all_depth.txt');
source_dir = '/home/enoon/DepthNet/Washington/rgbd-original';
output_dir = '/home/enoon/DepthNet/Washington/scaled_10_8bits';
for i=1:numel(filelist)
    in = fullfile(source_dir, filelist{i});
    out = fullfile(output_dir, filelist{i});
    convert_16to8(in,out);
end
