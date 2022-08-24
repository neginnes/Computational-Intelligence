clear all
close all
clc
load('All_data.mat')
%%
% Feature Extraction
Train_Features = [] ;
Number_of_trials = size(x_train,3);
Number_of_channels = size(x_train,2);
Fs = 100;
Train_Features = zeros(21*Number_of_channels + 0.5*Number_of_channels*(Number_of_channels+1) ,Number_of_trials);

for i = 1:Number_of_trials
    c = 1;
    for j = 1 : Number_of_channels
        Train_Features(c,i) = meanfreq(x_train(:,j,i),Fs);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = medfreq(x_train(:,j,i),Fs);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = obw(x_train(:,j,i),Fs);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = bandpower(x_train(:,j,i),Fs,[0.5,4]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = bandpower(x_train(:,j,i),Fs,[4,8]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = bandpower(x_train(:,j,i),Fs,[8,13]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = bandpower(x_train(:,j,i),Fs,[13,30]);
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = var(x_train(:,j,i));
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = skewness(x_train(:,j,i));
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        Train_Features(c,i) = kurtosis(x_train(:,j,i));
        c = c + 1;
    end
    for j = 1 : Number_of_channels
        for p = j : Number_of_channels
            Train_Features(c,i) = corr(x_train(:,j,i),x_train(:,p,i)) ;
            c = c + 1;
        end
    end
    for j = 1 : Number_of_channels
        h = histogram(x_train(:,j,i),10);
        for p = 1 : 10
            Train_Features(c,i) = h.Values(p);
            c = c + 1;
        end
    end
end
%%
% Normalization
[Normalized_Train_Features,xPS] = mapminmax(Train_Features) ;
save('Train_Features','Normalized_Train_Features','xPS')
%%
% Selecting features - part 1
Right_indices = find(y_train==1) ;
Left_indices = find(y_train==0) ;

Number_of_grouped_features = 1 ;
scores = [];
for i = 1 : size(Normalized_Train_Features,1)
    group_Train_Features = Normalized_Train_Features(i,:);
    scores = [scores,fisher_score(Number_of_grouped_features, Right_indices,Left_indices,group_Train_Features)];
end

Number_of_features = 20 ;
selected_features = [];
for i = 1:Number_of_features
    idx = find(scores == max(scores));
    selected_features = [selected_features,idx];
    scores(idx) = -1;
end
%%
Selected_Features = Normalized_Train_Features(selected_features,:);
Selected_Features_Indices = selected_features;
save('Selected_Features','Selected_Features','Selected_Features_Indices');
%% selecting features - part 2
Number_of_features = 7 ;
C = nchoosek(selected_features,Number_of_features);
Number_of_grouped_features = Number_of_features ;
scores = [];
for i = 1 : size(C,1)
    group_Train_Features = Normalized_Train_Features(C(i,1:Number_of_grouped_features),:);
    scores = [scores,fisher_score(Number_of_grouped_features,Right_indices,Left_indices,group_Train_Features)];
end
idx = find(scores == max(scores));
best_Train_features = Normalized_Train_Features(C(idx,1:Number_of_grouped_features),:);

%% ploting features
F = nchoosek(1:Number_of_features,3);
for i = 1:size(F,1)
    f = F(i,:);
    plot3(best_Train_features(f(1),Right_indices),best_Train_features(f(2),Right_indices),best_Train_features(f(3),Right_indices),'*r') ;
    hold on
    plot3(best_Train_features(f(1),Left_indices),best_Train_features(f(2),Left_indices),best_Train_features(f(3),Left_indices),'og') ;
    title(['Features ',num2str(f(1)),', ',num2str(f(2)),', ',num2str(f(3))],'interpreter','latex') ;
    hold off
    pause(1);
    
end
%% 1_hidden layer
f_act_array = {'hardlims','tansig','satlin','purelin','logsig'};
ACCMat = [];
for i = 1 : length(f_act_array)
    for N = 1 : 10
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
            net = patternnet(N,trainFcn,performFcn);
            net = train(net,TrainX,TrainY);
            f_act = f_act_array{i};
            net.layers{2}.transferFcn = f_act;
            predict_y = net(ValX);

            p_TrainY = net(TrainX);
            [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
            Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;

            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end

        ACCMat(i,N) = ACC/Number_of_trials ;
    end
end
Max_ACC1 = max(ACCMat,[],'all');
[f_act1,n1] = find(ACCMat == max(ACCMat,[],'all'));

%% 2  hidden layers
f_act_array = {'hardlims','tansig','satlin','purelin','logsig'};
ACCMat = [];
for i = 1 : length(f_act_array)
    for N1 = 1 : 5
        for N2 = 1 : 5
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

                % feedforwardnet, newff, paternnet
                % patternnet(hiddenSizes,trainFcn,performFcn)
                trainFcn = 'trainbr';
                performFcn = 'mse';
                net = patternnet([N1,N2],trainFcn,performFcn);
                net = train(net,TrainX,TrainY);
                f_act = f_act_array{i};
                net.layers{2}.transferFcn = f_act;
                net.layers{3}.transferFcn = f_act;
                predict_y = net(ValX);

                p_TrainY = net(TrainX);
                [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
                Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;

                predict_y = predict_y >= Thr ;

                ACC = ACC + length(find(predict_y==ValY)) ;
            end

            ACCMat(i,N1,N2) = ACC/Number_of_trials ;
        end
    end
end
Max_ACC2 = max(ACCMat,[],'all');
idx2 = find(ACCMat == max(ACCMat,[],'all'));
[I21,I22,I23] = ind2sub(size(ACCMat),idx2);

%% 3  hidden layers
f_act_array = {'hardlims','tansig','satlin','purelin','logsig'};
ACCMat = [];
for i = 1 : length(f_act_array)
    for N1 = 1 : 3
        for N2 = 1 : 3
            for N3 = 1 : 3
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

                    % feedforwardnet, newff, paternnet
                    % patternnet(hiddenSizes,trainFcn,performFcn)
                    trainFcn = 'trainbr';
                    performFcn = 'mse';
                    net = patternnet([N1,N2,N3],trainFcn,performFcn);
                    net = train(net,TrainX,TrainY);
                    f_act = f_act_array{i};
                    net.layers{2}.transferFcn = f_act;
                    net.layers{3}.transferFcn = f_act;
                    net.layers{4}.transferFcn = f_act;
                    predict_y = net(ValX);

                    p_TrainY = net(TrainX);
                    [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
                    Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;

                    predict_y = predict_y >= Thr ;

                    ACC = ACC + length(find(predict_y==ValY)) ;
                end

                ACCMat(i,N1,N2,N3) = ACC/Number_of_trials ;
            end
        end
    end
end
Max_ACC3 = max(ACCMat,[],'all');
idx3 = find(ACCMat == max(ACCMat,[],'all'));
[I31,I32,I33,I34] = ind2sub(size(ACCMat),idx3);

%% 1_hidden layer best network
N = 6;
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
    net = patternnet(N,trainFcn,performFcn);
    net = train(net,TrainX,TrainY);
    net.layers{2}.transferFcn = f_act;
    predict_y = net(ValX);

    Thr = 0.5 ;

    predict_y = predict_y >= Thr ;

    ACC = ACC + length(find(predict_y==ValY)) ;
end

ACC = ACC/Number_of_trials 
%%
save('net1','net')

%% 2_hidden layer best network
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
save('net2','net')
%% 3_hidden layer best network
N1 = 3;
N2 = 1;
N3 = 3;
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
    net = patternnet([N1,N2,N3],trainFcn,performFcn);
    net = train(net,TrainX,TrainY);
    net.layers{2}.transferFcn = f_act;
    net.layers{3}.transferFcn = f_act;
    net.layers{4}.transferFcn = f_act;
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
save('net3','net')
%% Test 1 hidden layer
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
best_Test_features = Normalized_Test_Features(C(idx,1:Number_of_grouped_features),:);
%%
% Classification
load('net1') % Best network found in training step
TestX = best_Test_features ;
% TestY = y_test ; 

predict_y1 = net(TestX);
Thr = 0.5 ;
predict_y1 = predict_y1 >= Thr ;
%ACC = length(find(predict_y==TestY))/size(x_test,3);

%% Test 2 hidden layers

% Classification
load('net2') % Best network found in training step
TestX = best_Test_features ;
% TestY = y_test ; 

predict_y2 = net(TestX);
Thr = 0.5 ;
predict_y2 = predict_y2 >= Thr ;
%ACC = length(find(predict_y==TestY))/size(x_test,3);

%% Test 3 hidden layers

% Classification
load('net3') % Best network found in training step
TestX = best_Test_features ;
% TestY = y_test ; 

predict_y3 = net(TestX);
Thr = 0.5 ;
predict_y3 = predict_y3 >= Thr ;
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
spread = 0.9 ; % Best parameter found in training step
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
save('net4','net')
%% RBF - test
% Classification

load('net4') % Best network found in training step
TestX = best_Test_features ;
%TestY = Test_Label ; 

predict_y4 = net(TestX);

Thr = 0.5 ;
predict_y4 = predict_y4 >= Thr ;

%ACC = length(find(predict_y==TestY))/size(TestX,3);


%% saving the outputs
save('y1','predict_y1');
save('y2','predict_y2');
save('y3','predict_y3');
save('y4','predict_y4');