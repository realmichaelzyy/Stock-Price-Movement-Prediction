function [ accu ] = run_bagging( train_data, train_label )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    %ens_classifier = fitensemble(train_data, train_label, 'AdaBoostM1', 100,
    %'Tree'); % LogitBoost, Subspace, Bag ; 'AllPredictorCombinations' ; 'KNN'
    %'Discriminant'
bag_classifier = fitensemble(train_data, train_label, 'Bag', 500, 'Tree', 'type', 'classification');
bag_cv = crossval(bag_classifier, 'KFold', 5);
bag_error_rate = kfoldLoss(bag_cv);
accu = 1 - bag_error_rate;

% figure;
% plot(kfoldLoss(bag_cv,'mode','cumulative'));
% title('Classification Error vs. Iterations');
% xlabel('Number of trees');
% ylabel('Classification error');
% axis('LineWidth', 0.25);




end

