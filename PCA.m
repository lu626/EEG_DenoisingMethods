clear; clc; close all;

tic; % 记录处理开始时间

% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');

% 定义噪声幅度的范围
noise_amplitudes = 5:5:50;
channel_numbers = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];

% 初始化结果矩阵
SNR_results = zeros(length(noise_amplitudes), length(channel_numbers));
MSE_results = zeros(length(noise_amplitudes), length(channel_numbers));
NCC_results = zeros(length(noise_amplitudes), length(channel_numbers));

% --- 热图性能评估 ---
for c = 1:length(channel_numbers)
    num_channels = channel_numbers(c);
    if num_channels == 1
        eeg_data_multichannel = EEG_all_epochs(4514, :);
    else
        random_indices = randperm(size(EEG_all_epochs, 1), num_channels-1);
        eeg_data_multichannel = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)];
    end

    for i = 1:length(noise_amplitudes)
        noise_amp = noise_amplitudes(i);
        noisy_data = eeg_data_multichannel + noise_amp * randn(size(eeg_data_multichannel));

        [coeff, score, latent] = pca(noisy_data');
        num_components = find(cumsum(latent) / sum(latent) >= 0.95, 1);
        clean_data = (score(:, 1:num_components) * coeff(:, 1:num_components)')';

        signal = eeg_data_multichannel(1, :);
        noisy_signal = noisy_data(1, :);
        denoised_signal = clean_data(1, :);

        snr_val = 10 * log10(sum(signal.^2) / sum((noisy_signal - signal).^2));
        mse_val = mean((signal - denoised_signal).^2);
        ncc_val = sum(signal .* denoised_signal) / sqrt(sum(signal.^2) * sum(denoised_signal.^2));

        SNR_results(i, c) = snr_val;
        MSE_results(i, c) = mse_val;
        NCC_results(i, c) = ncc_val;
    end
end

% --- 绘制热图 ---
figure;
subplot(3, 1, 1); heatmap(channel_numbers, noise_amplitudes, SNR_results); title('SNR Heatmap'); ylabel('Noise Amplitude');
subplot(3, 1, 2); heatmap(channel_numbers, noise_amplitudes, MSE_results); title('MSE Heatmap'); ylabel('Noise Amplitude');
subplot(3, 1, 3); heatmap(channel_numbers, noise_amplitudes, NCC_results); title('NCC Heatmap'); xlabel('Channel Counts'); ylabel('Noise Amplitude');

% --- 找出SNR最佳通道数 ---
[~, best_channel_idx] = max(SNR_results(end, :));
best_channel_number = channel_numbers(best_channel_idx);

% --- 倒数20行测试 ---
test_indices = (size(EEG_all_epochs, 1) - 19):size(EEG_all_epochs, 1);
SNR_list = zeros(1, 20);
MSE_list = zeros(1, 20);
NCC_list = zeros(1, 20);

for k = 1:20
    row_idx = test_indices(k);
    if best_channel_number == 1
        data_multi = EEG_all_epochs(row_idx, :);
    else
        random_indices = randperm(size(EEG_all_epochs, 1), best_channel_number-1);
        data_multi = [EEG_all_epochs(row_idx, :); EEG_all_epochs(random_indices, :)];
    end

    noisy_multi = data_multi + 50 * randn(size(data_multi));
    [coeff_test, score_test, latent_test] = pca(noisy_multi');
    num_components_test = find(cumsum(latent_test) / sum(latent_test) >= 0.95, 1);
    clean_multi = (score_test(:, 1:num_components_test) * coeff_test(:, 1:num_components_test)')';

    signal = data_multi(1, :);
    noisy = noisy_multi(1, :);
    denoised = clean_multi(1, :);

    SNR_list(k) = 10 * log10(sum(signal.^2) / sum((noisy - signal).^2));
    MSE_list(k) = mean((signal - denoised).^2);
    NCC_list(k) = sum(signal .* denoised) / sqrt(sum(signal.^2) * sum(denoised.^2));
end

% --- 图示第4514行 ---
demo_data = [EEG_all_epochs(4514, :); EEG_all_epochs(randperm(size(EEG_all_epochs,1), best_channel_number-1), :)];
demo_noisy = demo_data + 50 * randn(size(demo_data));
[coeff_demo, score_demo, latent_demo] = pca(demo_noisy');
num_components_demo = find(cumsum(latent_demo) / sum(latent_demo) >= 0.95, 1);
demo_clean = (score_demo(:, 1:num_components_demo) * coeff_demo(:, 1:num_components_demo)')';

figure;
subplot(3, 1, 1);
plot(EEG_all_epochs(4514, :));
title('Original EEG Signal (Row 4514)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(demo_noisy(1, :));
title(['Noisy Signal (Amplitude 50, Channels: ' num2str(best_channel_number) ')']);
ylabel('Amplitude');

subplot(3, 1, 3);
plot(demo_clean(1, :));
title('PCA Denoised EEG Signal');
xlabel('Time (points)');
ylabel('Amplitude');

% --- 显示最终平均结果 ---
fprintf('\n=== Final Results over Last 20 Rows ===\n');
fprintf('Best Channel Number: %d\n', best_channel_number);
fprintf('Mean SNR: %.2f dB\n', mean(SNR_list));
fprintf('Mean MSE: %.4f\n', mean(MSE_list));
fprintf('Mean NCC: %.4f\n', mean(NCC_list));

% --- 参数量与处理时间统计（PCA无可训练参数） ---
param_count = 0;
total_processing_time = toc;

fprintf('PCA Parameter Count: %d\n', param_count);
fprintf('PCA Total Processing Time: %.2f seconds\n', total_processing_time);
