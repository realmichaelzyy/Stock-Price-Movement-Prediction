function [ accu, opt_c, opt_g, opt_svm ] = run_svm( train_data, train_label )
% run_svm runs svm on input data under a variety of C and Gamma values.

% == Grid Search ==
% RBF kernel function
C = [2^(-5) 2^(-3) 2^(-2) 2^(-1) 1 2 4 8 16 24];
G = [2^(-7) 2^(-6) 2^(-5) 2^(-4) 2^(-3) 2^(-2) 2^(-1) 1 2 4 8];
opt_svm = zeros(length(C), length(G));
for c = 1 : length(C)
    for g = 1 : length(G)
        opt_svm(c, g) = svmtrain(train_label, train_data, ['-h 0 -q -v 5 -c ' num2str(C(c)) ' -t 2 -g ' num2str(G(g))]);  
    end
end
[nn, ii] = max(opt_svm(:));
[opt_c, opt_g] = ind2sub(size(opt_svm), ii);
accu = nn / 100;
% if t == 10
%     figure;
%     imagesc(opt_svm); 
%     set(gca,'XTick',[]);
%     set(gca,'YTick',[]);
%     colormap Jet; % Winter; %Hot; %Jet Autumn; 
 
%end
end

