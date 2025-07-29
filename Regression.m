% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');

% 定义噪声幅度的范围
noise_amplitudes = 5:5:50; % 从5到50的步长为5

% 定义通道数量范围，确保包含第4514行
channel_numbers = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];

% 存储 SNR、MSE 和 NCC 结果
SNR_results = zeros(length(noise_amplitudes), length(channel_numbers));
MSE_results = zeros(length(noise_amplitudes), length(channel_numbers));
NCC_results = zeros(length(noise_amplitudes), length(channel_numbers));

% ============================
% 开始并行计算
% ============================

parfor c = 1:length(channel_numbers)
    num_channels = channel_numbers(c); % 当前通道数量
    % 确保包含第4514行
    if num_channels == 1
        eeg_data_multichannel = EEG_all_epochs(4514, :); % 只选择第4514行
    else
        random_indices = randperm(size(EEG_all_epochs, 1), num_channels - 1); 
        eeg_data_multichannel = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)]; % 包含第4514行的多通道数据
    end

    % 临时存储每个通道数下每种噪声幅度的 SNR、MSE 和 NCC
    temp_SNR = zeros(length(noise_amplitudes), 1);
    temp_MSE = zeros(length(noise_amplitudes), 1);
    temp_NCC = zeros(length(noise_amplitudes), 1);

    for i = 1:length(noise_amplitudes)
        noise_amplitude = noise_amplitudes(i);
        % 添加噪声（每个通道单独加噪）
        noisy_eeg_multichannel = eeg_data_multichannel + noise_amplitude * randn(size(eeg_data_multichannel));

        % 回归降噪：岭回归（Ridge Regression）
        X = noisy_eeg_multichannel'; % 转置为 [时间 × 通道] 格式
        y = eeg_data_multichannel(1, :)'; % 使用原始信号（第4514行）作为目标
        lambda = 1;  % 正则化参数（可以调整）
        b_ridge = (X' * X + lambda * eye(size(X, 2))) \ (X' * y); % 求解岭回归系数 b
        denoised_signal = (X * b_ridge)'; % 通过岭回归系数进行预测，并转置为行向量

        % 计算 SNR、MSE 和 NCC
        signal = eeg_data_multichannel(1, :); % 原始信号（第4514行）
        noisy_signal = noisy_eeg_multichannel(1, :); % 加噪信号

        % 计算信噪比 (SNR)
        noise_signal = noisy_signal - signal;
        signal_power = sum(signal.^2) / length(signal);
        noise_power = sum(noise_signal.^2) / length(noise_signal);
        SNR = 10 * log10(signal_power / noise_power);

        % 计算均方误差 (MSE)
        MSE = mean((signal - denoised_signal).^2);

        % 计算归一化互相关系数 (NCC)
        NCC = sum(signal .* denoised_signal) / (sqrt(sum(signal.^2)) * sqrt(sum(denoised_signal.^2)));

        % 存储临时结果
        temp_SNR(i) = SNR;
        temp_MSE(i) = MSE;
        temp_NCC(i) = NCC;
    end

    % 将临时结果存储到全局变量
    SNR_results(:, c) = temp_SNR;
    MSE_results(:, c) = temp_MSE;
    NCC_results(:, c) = temp_NCC;
end

% ============================
% 生成热图
% ============================

% 创建一个新的图形
figure;
% 绘制 SNR 热图
subplot(3, 1, 1); % 第一行
heatmap(channel_numbers, noise_amplitudes, SNR_results);
title('SNR Heatmap');
ylabel('Noise Amplitude');
colorbar;

% 绘制 MSE 热图
subplot(3, 1, 2); % 第二行
heatmap(channel_numbers, noise_amplitudes, MSE_results);
title('MSE Heatmap');
ylabel('Noise Amplitude');
colorbar;

% 绘制 NCC 热图
subplot(3, 1, 3); % 第三行
heatmap(channel_numbers, noise_amplitudes, NCC_results);
title('NCC Heatmap');
xlabel('Channel Counts (RM)');
ylabel('Noise Amplitude');
colorbar;

% ============================
% 选择噪声幅度为50时，最佳通道数
% ============================
[~, best_channel_idx] = max(SNR_results(end, :)); % 最后一行对应噪声幅度为50的结果
best_channel_number = channel_numbers(best_channel_idx); % 最佳通道数

% 确保包含第4514行的通道数
if best_channel_number == 1
    eeg_data_multichannel_best = EEG_all_epochs(4514, :);
else
    random_indices = randperm(size(EEG_all_epochs, 1), best_channel_number - 1); 
    eeg_data_multichannel_best = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)]; % 包含第4514行的多通道数据
end

% 添加噪声（噪声幅度为50）
noisy_eeg_multichannel_best = eeg_data_multichannel_best + 50 * randn(size(eeg_data_multichannel_best));

% 使用岭回归进行降噪
X_best = noisy_eeg_multichannel_best'; 
y_best = eeg_data_multichannel_best(1, :)';
lambda_best = 1;
b_ridge_best = (X_best' * X_best + lambda_best * eye(size(X_best, 2))) \ (X_best' * y_best);
denoised_signal_best = (X_best * b_ridge_best)';

% 计算最终的 SNR、MSE 和 NCC
final_SNR = 10 * log10(sum(eeg_data_multichannel_best(1, :).^2) / sum((noisy_eeg_multichannel_best(1, :) - eeg_data_multichannel_best(1, :)).^2));
final_MSE = mean((eeg_data_multichannel_best(1, :) - denoised_signal_best).^2);
final_NCC = sum(eeg_data_multichannel_best(1, :) .* denoised_signal_best) / (sqrt(sum(eeg_data_multichannel_best(1, :).^2)) * sqrt(sum(denoised_signal_best.^2)));

% 输出最终结果
fprintf('Final SNR: %.4f dB\n', final_SNR);
fprintf('Final MSE: %.4f\n', final_MSE);
fprintf('Final NCC: %.4f\n', final_NCC);
figure;

% 原始信号
subplot(3, 1, 1);
plot(eeg_data_multichannel_best(1, :));
title('Original Signal (4514th row)');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 加噪信号
subplot(3, 1, 2);
plot(noisy_eeg_multichannel_best(1, :));
title('Noisy Signal (Noise amplitude = 50)');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 降噪信号
subplot(3, 1, 3);
plot(denoised_signal_best);
title('Denoised Signal (Using Ridge Regression)');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 参数量和处理时间统计（仅传统方法）
param_count = length(b_ridge_best);  % RM参数量即回归系数个数

% 重新执行一遍过程用于 timing
tic;
X_tmp = noisy_eeg_multichannel_best';
y_tmp = eeg_data_multichannel_best(1, :)';
b_tmp = (X_tmp' * X_tmp + lambda_best * eye(size(X_tmp,2))) \ (X_tmp' * y_tmp);
den_tmp = (X_tmp * b_tmp)';
dummySNR = 10 * log10(sum(y_tmp.^2) / sum((y_tmp' - den_tmp).^2)); % 触发运算
processing_time = toc;

% 输出参数量和处理耗时
fprintf('RM Parameter Count: %d\n', param_count);
fprintf('RM Processing Time: %.4f seconds\n', processing_time);
