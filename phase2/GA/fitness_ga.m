function score = fitness_ga(chromosome)
    features = [];
    load('Selected_Features')
    load('Train_Features')
    features_indices = find(chromosome==1);
    for i = features_indices
        features = [features; Selected_Features(i, :)];
    end
    Number_of_grouped_features = length(find(chromosome==1));
    load('All_data')
    Right_indices = find(y_train==1) ;
    Left_indices = find(y_train==0) ;
    N1 = length(Right_indices);
    N2 = length(Left_indices);
    Mu1 = mean(features(:,Right_indices),2);
    Mu2 = mean(features(:,Left_indices),2);
    Mu0 = mean(features,2);

    S1 = zeros(Number_of_grouped_features);
    S2 = zeros(Number_of_grouped_features);

    for i = 1 : N1
        S1 = S1 + (features(:,Right_indices(i))- Mu1)*(features(:,Right_indices(i))- Mu1)';
    end
    S1 = S1/N1;
    for i = 1 : N2
        S2 = S2 + (features(:,Left_indices(i))- Mu2)*(features(:,Left_indices(i))- Mu2)';
    end
    S2 = S2/N2;

    Sw = S1 + S2;
    Sb = (Mu1-Mu0)*(Mu1-Mu0)' + (Mu2-Mu0)*(Mu2-Mu0)';

    J = trace(Sb)/(trace(Sw)+1e-20);
    score =  J;
end