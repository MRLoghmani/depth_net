array=($(find . -name '*.png'))
for i in ${array[@]};do convert $i -define png:format=png8 $i; done
