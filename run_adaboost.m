function [ accu ] = run_adaboost( train_data, train_label )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

adaboost_classifier = fitensemble(train_data, train_label, 'AdaBoostM1', 500, 'Tree', 'type', 'classification');
adaboost_cv = crossval(adaboost_classifier, 'KFold', 5);
adaboost_error_rate = kfoldLoss(adaboost_cv);
accu = 1 - adaboost_error_rate;
% 
% figure;
% plot(kfoldLoss(adaboost_cv,'mode','cumulative'));
% title('Adaboost Classification Error vs. Iterations');
% xlabel('Number of trees');
% ylabel('Classification error');
% axis('LineWidth', 0.25);
end