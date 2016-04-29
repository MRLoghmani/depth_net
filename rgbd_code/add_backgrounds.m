filelist = importdata('1M_files.txt');
bg_paths = importdata('all_bgs.txt');
source_dir = '/home/enoon/DepthNet/Renderings/1MDataset';
output_dir1 = '/home/enoon/DepthNet/Renderings/1MDataset_kinect';
output_dir2 = '/home/enoon/DepthNet/Renderings/1MDataset_uniform';
bgs = {}; % this contains backgrounds already resized to appropriate size
for i=1:numel(bg_paths)
    bgs{i}.data = imresize(imread(bg_paths{i}), [256, 256]);
    bgs{i}.mean = get_center_mean(bgs{i}.data,3);
end
n_bgs = numel(bgs);
for i=1:numel(filelist)
    in = imread(fullfile(source_dir, filelist{i}));
    out1 = fullfile(output_dir1, filelist{i});
    out2 = fullfile(output_dir2, filelist{i});
    if exist(out2, 'file')
        continue
    end
    bg_i = randi(n_bgs);
    add_kinect_background(in, bgs{bg_i}.data, bgs{bg_i}.mean, out1);
    add_flat_background(in, out2);
    if mod(i,5000) == 0
      fprintf('%d out of %d\n',i,numel(filelist));
    end
end
