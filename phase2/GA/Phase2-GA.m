close all
clear all
clc
load('All_data.mat')
load('Train_Features')
load('Selected_Features')
Number_of_trials = size(x_train,3);
Number_of_channels = size(x_train,2);
Fs = 100;
Right_indices = find(y_train==1) ;
Left_indices = find(y_train==0) ;
chromosome = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1];
indices = find(chromosome==1);
best_Train_features = Selected_Features(indices,:);
%% 2_hidden layer best network
f_act_array = {'hardlims','tansig','satlin','purelin','logsig'};
ACCMat = [];
N1 = 2;
N2 = 3;
f_act = f_act_array{1};
ACC = 0 ;
% 5-fold cross-validation
K = 5;
for k = 1 : K
    train_indices = [1 : (k-1)*floor(Number_of_trials/K) , k*floor(Number_of_trials/K) + 1 : Number_of_trials] ;
    valid_indices = (k-1)*floor(Number_of_trials/K) + 1 : k*floor(Number_of_trials/K) ;

    TrainX = best_Train_features(:,train_indices) ;
    ValX = best_Train_features(:,valid_indices) ;
    TrainY = y_train(train_indices) ;
    ValY = y_train(valid_indices) ;
    % patternnet(hiddenSizes,trainFcn,performFcn)
    trainFcn = 'trainbr';
    performFcn = 'mse';
    net = patternnet([N1,N2],trainFcn,performFcn);
    net = train(net,TrainX,TrainY);
    net.layers{2}.transferFcn = f_act;
    net.layers{3}.transferFcn = f_act;
    predict_y = net(ValX);

    p_TrainY = net(TrainX);
    [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
    Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;

    predict_y = predict_y >= Thr ;
    length(find(predict_y==ValY))
    ACC = ACC + length(find(predict_y==ValY)) ;
end

ACC = ACC/Number_of_trials 
%%
save('net7','net')
%% Test 2 hidden layer
% Feature Extraction
Test_Features = [] ;
Number_of_trials = size(x_test,3);
Number_of_channels = size(x_test,2);
Fs = 100;
Test_Features = zeros(20*Number_of_channels + 0.5*Number_of_channels*(Number_of_channels+1) ,Number_of_trials);

for i = 1:Number_of_trials 
    c = 1;
    for j = 1 : Number_of_channels
        Test_Features(c,i) = meanfreq(x_test(:,j,i),Fs);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = medfreq(x_test(:,j,i),Fs);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = obw(x_test(:,j,i),Fs);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = bandpower(x_test(:,j,i),Fs,[0.5,4]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = bandpower(x_test(:,j,i),Fs,[4,8]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = bandpower(x_test(:,j,i),Fs,[8,13]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = bandpower(x_test(:,j,i),Fs,[13,30]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = var(x_test(:,j,i));
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = skewness(x_test(:,j,i));
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Test_Features(c,i) = kurtosis(x_test(:,j,i));
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        for p = j : Number_of_channels
            Test_Features(c,i) = corr(x_test(:,j,i),x_test(:,p,i)) ;
            c = c + 1;
        end
    end
    for j = 1 : Number_of_channels
        h = histogram(x_train(:,j,i),10);
        for p = 1 : 10
            Test_Features(c,i) = h.Values(p);
            c = c + 1;
        end
    end
end
% Normalization
Normalized_Test_Features = mapminmax('apply',Test_Features,xPS) ;
selected_Test_features = Normalized_Test_Features(Selected_Features_Indices,:);
best_Test_features = selected_Test_features(indices,:);
%%
% Classification
load('net7') % Best network found in training step
TestX = best_Test_features ;
% TestY = y_test ; 

predict_y7 = net(TestX);
Thr = 1 ;
predict_y7 = predict_y7 >= Thr ;
%ACC = length(find(predict_y==TestY))/size(x_test,3);

%% RBF network
ACCMat = [];
spreadMat = [0.01,0.1,0.5,0.9,1.5,2,2.5,3,5,10,15] ;
NMat = [5,10,15,20,25] ;
for s = 1:length(spreadMat)
    spread = spreadMat(s) ;
    for n = 1:length(NMat) 
        Maxnumber = NMat(n) ;
        ACC = 0 ;
        % 5-fold cross-validation
        K = 5;
        for k=1:K
            train_indices = [1 : (k-1)*floor(Number_of_trials/K) , k*floor(Number_of_trials/K) + 1 : Number_of_trials] ;
            valid_indices = (k-1)*floor(Number_of_trials/K) + 1 : k*floor(Number_of_trials/K) ;
            
            TrainX = best_Train_features(:,train_indices) ;
            ValX = best_Train_features(:,valid_indices) ;
            TrainY = y_train(train_indices) ;
            ValY = y_train(valid_indices) ;
            

            net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
            predict_y = net(ValX);
            
            p_TrainY = net(TrainX);
            [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
            Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end
        ACCMat(s,n) = ACC/Number_of_trials ;
    end
end
Max_ACC4 = max(ACCMat,[],'all');
idx4 = find(ACCMat == max(ACCMat,[],'all'));
[I41,I42] = ind2sub(size(ACCMat),idx4);

%% RBF - best network
spread = 5 ; % Best parameter found in training step
Maxnumber = 15 ; % Best parameter found in training step
ACC = 0 ;
% 5-fold cross-validation
K = 5;
for k=1:K
    train_indices = [1 : (k-1)*floor(Number_of_trials/K) , k*floor(Number_of_trials/K) + 1 : Number_of_trials] ;
    valid_indices = (k-1)*floor(Number_of_trials/K) + 1 : k*floor(Number_of_trials/K) ;

    TrainX = best_Train_features(:,train_indices) ;
    ValX = best_Train_features(:,valid_indices) ;
    TrainY = y_train(train_indices) ;
    ValY = y_train(valid_indices) ;


    net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
    predict_y = net(ValX);

    p_TrainY = net(TrainX);
    [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
    Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
    predict_y = predict_y >= Thr ;

    ACC = ACC + length(find(predict_y==ValY)) ;
end
ACC = ACC/Number_of_trials 
%%
save('net8','net')
%% RBF - test
% Classification

load('net8') % Best network found in training step
TestX = best_Test_features ;
%TestY = Test_Label ; 

predict_y8 = net(TestX);

Thr = 0.4620 ;
predict_y8 = predict_y8 >= Thr ;
%ACC = length(find(predict_y==TestY))/size(TestX,3);

%%  saving the outpts
save('y7','predict_y7');
save('y8','predict_y8');


