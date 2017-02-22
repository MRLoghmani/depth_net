clear
close all
rand('state', 0);

% Load the datasplit and distance matrix
orig = load('../data/jhuit_original_kernel.mat');
norm = load('../data/jhuit_normalized_kernel.mat');
rgb = load('../data/jhuit_rgb_kernel.mat');

% Create the training data and testing data
% In the demo we use split 1 of the flower dataset
NK=3;
Ktrain = zeros(7349, 7349, NK);
Ktrain(:,:,1)= orig.train_kernel; %exp(-D_colourgc(trsplit, trsplit)/mean(mean(D_colourgc(trsplit, trsplit))));
Ktrain(:,:,2)=norm.train_kernel; %exp(-D_hog(trsplit, trsplit)/mean(mean(D_hog(trsplit, trsplit))));
Ktrain(:,:,3)=orig.train_kernel;
Ktrain = single(Ktrain);

Ktest = zeros(7349, 7349, NK);
Ktest(:,:,1)= orig.test_kernel;  %exp(-D_colourgc(trsplit, trsplit)/mean(mean(D_colourgc(trsplit, trsplit))));
Ktest(:,:,2)=norm.test_kernel; %exp(-D_hog(trsplit, trsplit)/mean(mean(D_hog(trsplit, trsplit))));
Ktest(:,:,3)=orig.train_kernel;
Ktest = single(Ktest);

%Ktrain = zeros(7349, 7349, NK);
%Ktrain(:,:,1)=orig.train_kernel; %exp(-D_colourgc(trsplit, trsplit)/mean(mean(D_colourgc(trsplit, trsplit))));
%Ktrain(:,:,2)=norm.train_kernel; %exp(-D_hog(trsplit, trsplit)/mean(mean(D_hog(trsplit, trsplit))));
%Ktrain(:,:,3)=rgb.train_kernel;
%Ktrain = single(Ktrain);

%Ktest = zeros(7349, 7349, NK);
%Ktest(:,:,1)=orig.test_kernel;  %exp(-D_colourgc(trsplit, trsplit)/mean(mean(D_colourgc(trsplit, trsplit))));
%Ktest(:,:,2)=norm.test_kernel; %exp(-D_hog(trsplit, trsplit)/mean(mean(D_hog(trsplit, trsplit))));
%Ktest(:,:,3)=rgb.train_kernel;
%Ktest = single(Ktest);

% gernate the labels of the dataset
Ytrain = orig.train_labels+1;
Ytest  = orig.test_labels+1;
  
% Parameters for Obscure
C                  = 10;
model_zero         = model_init();
model_zero.n_cla   = 49;
model_zero.T1      = 100;   % Maximum number of epochs for the online stage
model_zero.T2      = 100;   % Maximum number of epochs for the batch stage
model_zero.p       = 1.05;
model_zero.lambda  = 1/(C*numel(Ytrain));
model_zero.eta     = 2;

model_zero.step   = 10*numel(Ytrain);
options.eachRound = @obscure_test;
options.Ktest     = Ktest;
options.Ytest     = Ytest;

model = k_obscure_online_train(Ktrain, Ytrain, model_zero, options);

semilogx(model.test,model.obj,'x-')
xlabel('Number of updates')
ylabel('Objective function')
grid
