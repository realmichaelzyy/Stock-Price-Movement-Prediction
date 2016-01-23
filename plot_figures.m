bagging = 100*performance_bagging(1, :);
svm = 100*performance_svm(1, :);
adaboost = 100*performance_adaboost(1, :);
baseline_s = 100*performance_adaboost(2, :);
days = time_window;

figure;
plot(days, bagging, 'r');
hold on;
plot(days, svm, 'b');
hold on;
plot(days, adaboost, 'g');
hold on;
plot(days, baseline_s, 'c');
title('Prediction Accuracy vs. Time Window');
xlabel('Time Window (Days)')
ylabel('Cross Validation Accuracy (%)');
legend('Bootstrap Aggregation','SVM', 'Adaboost', 'Baseline');

baseline_percentage_difference = zeros(4, length(bagging));
baseline_percentage_difference(1, :) = 100 * (bagging - baseline_s) ./ baseline_s;
baseline_percentage_difference(2, :) = 100 * (svm - baseline_s) ./ baseline_s;
baseline_percentage_difference(3, :) = 100 * (adaboost - baseline_s) ./ baseline_s;
baseline_percentage_difference(4, :) = days;

figure;
scatter(days, baseline_percentage_difference(1, :), 'r', 'filled');
hold on;
scatter(days, baseline_percentage_difference(2, :), 'b', 'filled');
hold on;
scatter(days, baseline_percentage_difference(3, :), 'g', 'filled');
hold on;
title('Model Performance Analysis');
xlabel('Time Window (Days)')
ylabel('Accuracy Percentage Gain (%)');
legend('Bootstrap Aggregation','SVM', 'Adaboost');

figure
h1 = scatter(down_labels(:, 2), down_labels(:, 3),'o');
hold on
h2 = scatter(up_labels(:, 2), up_labels(:, 3),'x');
title('RSI vs. %DS')
xlabel('RSI')
ylabel('%DS')

figure
h1 = scatter3(cool_1(:, 4), cool_1(:, 3), cool_1(:, 2),'o');
%legend('Up')
hold on
h2 = scatter3(cool_2(:, 4), cool_2(:, 3), cool_2(:, 2),'x');

%legend('Down')
xlabel('SMAVG(5)')
ylabel('%DS')
zlabel('RSI')


