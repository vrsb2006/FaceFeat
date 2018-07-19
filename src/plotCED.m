% plot the cumulative error distribution curve
error = load('error_49.txt');
error = sort(error);
plot(error, [1:length(error)]/length(error), 'm-', 'linewidth', 2);
xlim([0 0.3]);
ylim([0 1]);
grid on;
legend({'indoor+outdoor images'});
xlabel('Normalised error by inter-ocular distance');
ylabel('Propotion of test images');
title('CED curve on 49 point annotations on ibug 300w database')