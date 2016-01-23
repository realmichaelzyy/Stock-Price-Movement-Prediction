function [ accu ] = run_quad( train_data, train_label )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
quad_classifier = fitcdiscr(train_data, train_label, 'DiscrimType', 'quadratic'); 
quad_cv = crossval(quad_classifier, 'KFold', 5);
quad_error_rate = kfoldLoss(quad_cv);
accu = 1 - quad_error_rate;
end

