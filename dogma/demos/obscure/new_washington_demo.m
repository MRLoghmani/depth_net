
%clear
close all
rand('state', 0);

fprintf('Will load split %d', SPLIT)
% Load the datasplit and distance matrix
K = {
strcat('../data/kernels/washington_rgb__',num2str(SPLIT)), 
strcat('../data/kernels/washington_20160823-142643-90d2_original_', num2str(SPLIT)), 
};

% gernate the labels of the dataset
Ytrain = h5read(K{1}, '/train_labels')' + 1;
Ytest  = h5read(K{1}, '/test_labels')' + 1;
trainS = size(Ytrain, 2)
testS = size(Ytest, 2)
% Create the training data and testing data
NK=numel(K)
Ktrain = zeros(trainS, trainS, NK);
Ktest = zeros(trainS, testS, NK);
for i=1:NK
    Ktrain(:,:,i) = h5read(K{i}, '/train_kernel')';
    Ktest(:,:,i) = h5read(K{i}, '/test_kernel')';
end

Ktrain = single(Ktrain);
Ktest = single(Ktest);

disp 'Finished loading kernels'; 
 
% Parameters for OBSCURE
%C                  = 1 %1000: 89.66 10: 90.33 0.1: 88
model_zero         = model_init();
model_zero.n_cla   = 51;
model_zero.T1      = 100;   % Maximum number of epochs for the online stage
model_zero.T2      = 300;   % Maximum number of epochs for the batch stage
model_zero.p       = P; % 1.05; % 1.05;
model_zero.lambda  = 1/(C*numel(Ytrain));
model_zero.eta     = 2;

model_zero.step   = 10*numel(Ytrain);
options.eachRound = @obscure_test;
options.Ktest     = Ktest;
options.Ytest     = Ytest;

model = k_obscure_train(Ktrain, Ytrain, model_zero, options);
acc = model.acc2(end);
semilogx(model.test,model.obj,'x-')
xlabel('Number of updates')
ylabel('Objective function')
grid