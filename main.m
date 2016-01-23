clear;
%time_window = [1:19, 20:20:400];
file_name = 'Z:\Fall 2015\CS 260\Term Project\Alphabet Inc 7-feature data.csv';
data = csvread(file_name);
[M, D]=size(data);
time_window = [1:4 5:5:90];
%results = zeros(length(time_window), 5);  % log reg, d tree, svm, ens, knn
% The first row in performance matrix is the prediction accuracy
% and the second row is the stock's baseline performance, the
% higher of the up and down label percentage. Each column is a different
% time window, with the first 5 each being 1 to 5 days, and starting from
% the 6th column increment by 5, so 1 2 3 4 5 10 15 20 25 ... 60
L = length(time_window);
opt_c_set = zeros(1, L);
opt_g_set = zeros(1, L);
performance_svm = zeros(2, L);
performance_bagging = zeros(2, L);
performance_adaboost = zeros(2, L);
performance_quad = zeros(2, L);
for t = 1 : length(time_window)
    %Close,%B,%DS(5),%DSS(3),Price Movement
    %Price,BollW,%B,SMAVG(5) on Close,SMAVG(20) on Close,RSI(14) On Close,%DS(5),%DSS(3),30-day Price Difference
    % Close,UBB(2),BollMA(20),LBB(2),BollW,%B,SMAVG(5) on Close,CMCI(13),positive DMI(14) on Close,negative DMI,ADX,Conv(9),Base(26),Lead1(26),Lead2(26),"MACD(12,26)",Sig(9),Diff,"Osc(5,15)",PTPS(0.02),ROC(1) on Close,RSI(14) on Close,%DS(5),%DSS(3),"TE UB(20,2)","TE LB(20,2)",WLPR(14),Periodic Total Return,Fisher Transform,"Transform MA(20,Simple)","MLR(14,0,0)",MLRUB(1),MLRLB(1),30-day Price Diff,30-day Mov Label
      %Alphabet Inc 1-day.csv';
    %NEWCAPEC 30-day 7-features.csv'; 8-feature 30-day Prediction Goog.csv
    %'Z:\Fall 2015\CS 260\Term Project\2 features Goog Bollinger.csv' features: Close,%B,Price Movement
    %6 features Goog Bollinger.csv

    % separate train and test
    %=== Create labels that correspond to specified time_window=====
    % The last column in data is the close price
    train_label = zeros(M, 1);
    for i = time_window(t)+1 : M
        train_label(i) = data(i - time_window(t), end) - data(i, end);
        if (train_label(i) == 0)  % If price did not change, view it as decrease
            train_label(i) = -1;
        end
    end
    train_label = train_label./abs(train_label);
    train_label = train_label(time_window(t)+1:end);
    train_data = data(time_window(t)+1 : end, :);
    
    % ============normalization of features===============
    [row, col] = size(train_data);
    sigma = std(train_data);
    mu = mean(train_data);
    for i = 1 : col
        train_data(:,i) = (train_data(:,i) - mu(i))./sigma(i);
    end
    % =============Computing Baseline Performance=========
    num_one = sum(train_label(:) == 1);
    num_neg = sum(train_label(:) == -1);
    baseline = max(num_one, num_neg)/numel(train_label);
    % ==================svm============
    [opt_accu, opt_c, opt_g, opt_svm] = run_svm(train_data, train_label);
    performance_svm(1, t) = opt_accu;
    performance_svm(2, t) = baseline;
    opt_c_set(t) = opt_c;
    opt_g_set(t) = opt_g;
    % =====================Bagging========================
    performance_bagging(1, t) = run_bagging(train_data, train_label);
    performance_bagging(2, t) = baseline;
    % =======================Adaboost=======================
    performance_adaboost(1, t) = run_adaboost(train_data, train_label);
    performance_adaboost(2, t) = baseline;
    % ====================Quadratic Discriminant Analysis============
    performance_quad(1, t) = run_quad(train_data, train_label);
    performance_quad(2, t) = baseline;
end
