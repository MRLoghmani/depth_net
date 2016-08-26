clear
close all
rand('state', 0);

% Load the datasplit and distance matrix
orig = load('../data/jhuit_original_kernel.mat');
norm = load('../data/jhuit_normalized_kernel.mat');

% Create the training data and testing data
% In the demo we use split 1 of the flower dataset
Ktrain = zeros(7349, 7349, 2);
Ktrain(:,:,1)=orig.train_kernel; %exp(-D_colourgc(trsplit, trsplit)/mean(mean(D_colourgc(trsplit, trsplit))));
Ktrain(:,:,2)=norm.train_kernel; %exp(-D_hog(trsplit, trsplit)/mean(mean(D_hog(trsplit, trsplit))));
Ktrain = single(Ktrain);

Ktest = zeros(7349, 7349, 2);
Ktest(:,:,1)=orig.test_kernel;  %exp(-D_colourgc(trsplit, trsplit)/mean(mean(D_colourgc(trsplit, trsplit))));
Ktest(:,:,2)=norm.test_kernel; %exp(-D_hog(trsplit, trsplit)/mean(mean(D_hog(trsplit, trsplit))));
Ktest = single(Ktest);

% gernate the labels of the dataset
Ytrain = orig.train_labels+1;
Ytest  = orig.test_labels+1;
  
% Parameters for UFO-MKL
C                  = 0.0000001;
model_zero         = model_init();
model_zero.n_cla   = 49;
model_zero.T       = 100;   % Number of epochs
model_zero.lambda  = 1/(C*numel(Ytrain));

model_zero.step   = 10*numel(Ytrain);
options.eachRound = @ufomkl_test;
options.Ktest     = Ktest;
options.Ytest     = Ytest;

model = k_ufomkl_multi_train(Ktrain, Ytrain, model_zero, options);

semilogx(model.test,model.obj,'x-')
xlabel('Number of updates')
ylabel('Objective function')
grid
