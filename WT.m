% ===============================
% 第一步：加载数据和初始化参数
% ===============================
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');

noise_amplitudes = 5:5:50;
channel_numbers = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];

SNR_results = zeros(length(noise_amplitudes), length(channel_numbers));
MSE_results = zeros(length(noise_amplitudes), length(channel_numbers));
NCC_results = zeros(length(noise_amplitudes), length(channel_numbers));

% ===============================
% 第二步：热图性能评估主循环
% ===============================
tic; % Start processing time timer

for c = 1:length(channel_numbers)
    num_channels = channel_numbers(c);
    if num_channels == 1
        base_data = EEG_all_epochs(4514, :);
    else
        rand_idx = randperm(size(EEG_all_epochs, 1), num_channels-1);
        base_data = [EEG_all_epochs(4514, :); EEG_all_epochs(rand_idx, :)];
    end

    for i = 1:length(noise_amplitudes)
        amp = noise_amplitudes(i);
        noisy_data = base_data + amp * randn(size(base_data));

        [C, L] = wavedec(noisy_data(1,:), 3, 'sym8');
        T = 0.1 * max(abs(C));
        C_denoised = wthresh(C, 's', T);
        denoised = waverec(C_denoised, L, 'sym8');

        signal = base_data(1, :);
        noisy = noisy_data(1, :);

        SNR_results(i, c) = 10*log10(sum(signal.^2)/sum((signal - noisy).^2));
        MSE_results(i, c) = mean((signal - denoised).^2);
        NCC_results(i, c) = sum(signal .* denoised) / sqrt(sum(signal.^2) * sum(denoised.^2));
    end
end

% ===============================
% 第三步：绘制热图
% ===============================
figure;
subplot(3,1,1); heatmap(channel_numbers, noise_amplitudes, SNR_results);
title('SNR Heatmap'); ylabel('Noise Amplitude');
subplot(3,1,2); heatmap(channel_numbers, noise_amplitudes, MSE_results);
title('MSE Heatmap'); ylabel('Noise Amplitude');
subplot(3,1,3); heatmap(channel_numbers, noise_amplitudes, NCC_results);
title('NCC Heatmap'); ylabel('Noise Amplitude'); xlabel('Channel Counts');

% ===============================
% 第四步：选择噪声50时最佳通道数
% ===============================
[~, best_c_idx] = max(SNR_results(end, :));
best_channels = channel_numbers(best_c_idx);

% ===============================
% 第五步：倒数20行平均性能评估
% ===============================
test_rows = (size(EEG_all_epochs,1) - 19):size(EEG_all_epochs,1);
SNR_list = zeros(1,20);
MSE_list = zeros(1,20);
NCC_list = zeros(1,20);

for k = 1:20
    row = test_rows(k);
    if best_channels == 1
        data = EEG_all_epochs(row, :);
    else
        rand_idx = randperm(size(EEG_all_epochs,1), best_channels-1);
        data = [EEG_all_epochs(row, :); EEG_all_epochs(rand_idx, :)];
    end

    noisy = data + 50 * randn(size(data));
    [C, L] = wavedec(noisy(1,:), 3, 'sym8');
    T = 0.1 * max(abs(C));
    C_denoised = wthresh(C, 's', T);
    denoised = waverec(C_denoised, L, 'sym8');

    clean = data(1,:);
    noisy1 = noisy(1,:);

    SNR_list(k) = 10*log10(sum(clean.^2)/sum((noisy1 - clean).^2));
    MSE_list(k) = mean((clean - denoised).^2);
    NCC_list(k) = sum(clean .* denoised) / sqrt(sum(clean.^2) * sum(denoised.^2));
end

% ===============================
% 第六步：绘图（使用第4514行）
% ===============================
demo_clean = EEG_all_epochs(4514, :);
if best_channels == 1
    data = demo_clean;
else
    rand_idx = randperm(size(EEG_all_epochs,1), best_channels-1);
    data = [demo_clean; EEG_all_epochs(rand_idx, :)];
end
noisy = data + 50 * randn(size(data));
[C, L] = wavedec(noisy(1,:), 3, 'sym8');
T = 0.1 * max(abs(C));
C_denoised = wthresh(C, 's', T);
denoised = waverec(C_denoised, L, 'sym8');

figure;
subplot(3,1,1);
plot(demo_clean); title('Original Signal (Row 4514)');
subplot(3,1,2);
plot(noisy(1,:)); title('Noisy Signal');
subplot(3,1,3);
plot(denoised); title('Denoised Signal (Wavelet)');

% ===============================
% 第七步：输出最终平均指标
% ===============================
fprintf('\n=== Final Results (WT, average of last 20 rows) ===\n');
fprintf('Best Channel Number: %d\n', best_channels);
fprintf('SNR: %.2f dB\n', mean(SNR_list));
fprintf('MSE: %.4f\n', mean(MSE_list));
fprintf('NCC: %.4f\n', mean(NCC_list));

% ===============================
% 第八步：参数量和处理耗时输出
% ===============================
param_count = 0;  % Wavelet无训练参数
processing_time = toc;

fprintf('Parameter Count: %d\n', param_count);
fprintf('Processing Time: %.4f seconds\n', processing_time);
