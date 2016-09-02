% Demo for OBSCURE algorithm on Oxford Flower dataset, this is demo is to
% replicate the experiments conducted in Section 4.1 of the CVPR paper.
% Please read the README first and download the data from the dataset
% website.
%
%   References:
%     - Orabona, F., Jie, L., and Caputo, B. (2010).
%       Online-Batch Strongly Convex Multi Kernel Learning.
%       Proceedings of the 23rd IEEE Conference on Computer Vision and
%       Pattern Recognition.

%    This file is part of the DOGMA library for MATLAB.
%    Copyright (C) 2009-2011, Francesco Orabona
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%    Contact the authors: francesco [at] orabona.com
%                         jluo      [at] idiap.ch

clear
close all
rand('state', 0);

% Load the datasplit and distance matrix
K = {
'../data/kernels/jhuit_rgb_0'), 
'../data/kernels/jhuit_depth_norm_0'), 
};

% Create the training data and testing data
NK=numel(K)
Ktrain = zeros(7349, 7349, NK);
Ktest = zeros(7349, 7349, NK);
for i=1:NK
    Ktrain(:,:,i) = h5read(K{i}, '/train_kernel');
    Ktest(:,:,i) = h5read(K{i}, '/test_kernel');
end

Ktrain = single(Ktrain);
Ktest = single(Ktest);

disp 'Finished loading kernels'; 

% gernate the labels of the dataset
Ytrain = h5read(K{1}, '/train_labels') + 1;
Ytest  = h5read(K{1}, '/test_labels') + 1;
  
% Parameters for OBSCURE
C                  = 100; %1000: 89.66 10: 90.33 0.1: 88
model_zero         = model_init();
model_zero.n_cla   = 49;
model_zero.T1      = 100;   % Maximum number of epochs for the online stage
model_zero.T2      = 300;   % Maximum number of epochs for the batch stage
model_zero.p       = 1.01; % 1.05;
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