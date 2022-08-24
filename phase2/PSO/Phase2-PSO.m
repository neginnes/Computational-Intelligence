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
%% feature selecting with PSO algorithm
N = size(Selected_Features,1);
Number_of_particles = 15;
X = [];
for i = 1 : Number_of_particles
    particle_indices = randperm(N,15);
    X = [X;particle_indices];
end
X_local = X;

Number_of_grouped_features = 15 ;
fitness = 0;
Max_iterations = 1000;
V = zeros(size(X));
iteration = 0;
while (fitness < 0.1 && iteration <= Max_iterations)
    scores = [];
    for i = 1 : size(X,1)
        group_Train_Features = Selected_Features(X(i,1:Number_of_grouped_features),:);
        scores = [scores,fisher_score(Number_of_grouped_features,Right_indices,Left_indices,group_Train_Features)];
    end
    idx = find(scores == max(scores));
    X_global = ones(Number_of_particles,1) * X(idx(1),:);
    fitness = max(scores);
    
    next_scores = [];
    alpha = rand(1);
    beta1 = rand(1);
    beta2 = rand(1);
    V = alpha*V + beta1*(X_local - X) + beta2*(X_global - X);
    next_X = min(max(X + round(V),ones(size(X))),N*ones(size(X))); 
    for i = 1 : size(next_X,1)
        group_Train_Features = Selected_Features(next_X(i,1:Number_of_grouped_features),:);
        next_scores = [next_scores,fisher_score(Number_of_grouped_features,Right_indices,Left_indices,group_Train_Features)];
    end
    for i = 1 : Number_of_particles
        if (next_scores(i) > scores(i))
            X_local(i,:) = next_X(i,:);
        end
    end
    if (max(next_scores) > max(scores))
        idx = find(next_scores == max(next_scores));
        X_global = ones(Number_of_particles,1) * next_X(idx(1),:);
    elseif (max(next_scores) <= max(scores))
        idx = find(scores == max(scores));
        X_global = ones(Number_of_particles,1) * X(idx(1),:);
    end
    X = next_X;
    fitness = max(max(scores),max(next_scores));
    process_percentage = 100*iteration/Max_iterations
    iteration = iteration + 1;
end

%%
indices = X(1,:);
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
save('net5','net')
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
load('net5') % Best network found in training step
TestX = best_Test_features ;
% TestY = y_test ; 

predict_y5 = net(TestX);
Thr = 1 ;
predict_y5 = predict_y5 >= Thr ;
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
            
            Thr = 0.5 ;
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
spread = 1.5 ; % Best parameter found in training step
Maxnumber = 5 ; % Best parameter found in training step
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

    Thr = 0.5 ;
    predict_y = predict_y >= Thr ;

    ACC = ACC + length(find(predict_y==ValY)) ;
end
ACC = ACC/Number_of_trials 
%%
save('net6','net')
%% RBF - test
% Classification

load('net6') % Best network found in training step
TestX = best_Test_features ;
%TestY = Test_Label ; 

predict_y6 = net(TestX);

Thr = 0.5 ;
predict_y6 = predict_y6 >= Thr ;

%ACC = length(find(predict_y==TestY))/size(TestX,3);

%%  saving the outpts
save('y5','predict_y5');
save('y6','predict_y6');

