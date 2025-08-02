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

        % 保持你原有的计算方式不变
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
subplot(3, 1, 3); heatmap(channel_numbers, noise_amplitudes, NCC_results); title('NCC Heatmap'); xlabel('Channel Counts (PCA)'); ylabel('Noise Amplitude');

% --- 找出SNR最佳通道数（A=50） ---
[~, best_channel_idx] = max(SNR_results(end, :));
best_channel_number = channel_numbers(best_channel_idx);



% --- 图示第4514行，并计算单次指标 ---
demo_data = [EEG_all_epochs(4514, :); EEG_all_epochs(randperm(size(EEG_all_epochs,1), best_channel_number-1), :)];
demo_noisy = demo_data + 50 * randn(size(demo_data));
[coeff_demo, score_demo, latent_demo] = pca(demo_noisy');
num_components_demo = find(cumsum(latent_demo) / sum(latent_demo) >= 0.95, 1);
demo_clean = (score_demo(:, 1:num_components_demo) * coeff_demo(:, 1:num_components_demo)')';

signal = demo_data(1, :);
noisy  = demo_noisy(1, :);
denoised = demo_clean(1, :);

SNR_single = 10 * log10(sum(signal.^2) / sum((noisy - signal).^2));
MSE_single = mean((signal - denoised).^2);
NCC_single = sum(signal .* denoised) / sqrt(sum(signal.^2) * sum(denoised.^2));

figure;
subplot(3, 1, 1);
plot(EEG_all_epochs(4514, :));
title('Original EEG Signal (Row 4514)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(noisy);
title(['Noisy Signal (Amplitude 50, Channels: ' num2str(best_channel_number) ')']);
ylabel('Amplitude');

subplot(3, 1, 3);
plot(denoised);
title('PCA Denoised EEG Signal');
xlabel('Time (points)');
ylabel('Amplitude');

% --- 显示“单次”结果 ---
fprintf('\n=== Final Result on Row 4514 ===\n');
fprintf('Best Channel Number: %d\n', best_channel_number);
fprintf('SNR: %.2f dB\n', SNR_single);
fprintf('MSE: %.4f\n', MSE_single);
fprintf('NCC: %.4f\n', NCC_single);

% --- 参数量与处理时间统计（PCA无可训练参数） ---
param_count = 0;
total_processing_time = toc;

fprintf('PCA Parameter Count: %d\n', param_count);
fprintf('PCA Total Processing Time: %.2f seconds\n', total_processing_time);
