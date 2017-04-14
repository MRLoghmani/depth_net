filelist = textread('Total_list.txt','%s',4000000);
for i=1:numel(filelist)%
	imwrite(imread(filelist{i}), filelist{i});
    if mod(i,5000) == 0
      fprintf('%d out of %d\n',i,numel(filelist));
    end
end
