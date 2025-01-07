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

% 定义改进后的 CCA 降噪函数
function clean_data = cca_denoise(noisy_data)
    [N, T] = size(noisy_data);  % N 为通道数，T 为时间点数
    
    % 数据中心化：每个通道的均值减去
    noisy_data_centered = noisy_data - mean(noisy_data, 2);
    
    % 如果只有一个通道，直接返回
    if N == 1
        clean_data = noisy_data;
        return;
    end
    
    % 计算协方差矩阵
    R = cov(noisy_data_centered');  % 使用内置的cov函数来计算协方差矩阵，行是通道数，列是时间点数
    
    % 正则化协方差矩阵，确保其可逆
    epsilon = 1e-6;  % 正则化参数
    R = R + epsilon * eye(N);  % 对协方差矩阵进行正则化，避免秩不足问题
    
    % 对协方差矩阵进行奇异值分解（SVD）
    [U, S, V] = svd(R);  % SVD分解
    singular_values = diag(S);  % 获取奇异值
    
    % 选择主要成分，保留前面几个最大的奇异值
    threshold = 0.1 * max(singular_values);  % 设置阈值，保留大于阈值的成分
    num_components = sum(singular_values > threshold);  % 根据阈值选择主成分的数量
    num_components = max(1, min(num_components, floor(N/2)));  % 确保选择的成分数量合理
    
    % 投影矩阵：选取最重要的成分
    W = U(:, 1:num_components);  % 选取奇异值大的成分
    
    % 重构信号
    Y = W' * noisy_data_centered;  % 投影数据到新的子空间
    X_clean = W * Y;  % 从降维后的数据重构信号
    
    % 将数据反中心化，恢复原始数据的均值
    clean_data = X_clean + mean(noisy_data, 2);
end

% 主循环处理
for c = 1:length(channel_numbers)
    num_channels = channel_numbers(c);
    
    % 确保包含第4514行
    if num_channels == 1
        eeg_data_multichannel = EEG_all_epochs(4514, :);
    else
        random_indices = randperm(size(EEG_all_epochs, 1), num_channels-1);
        eeg_data_multichannel = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)];
    end
    
    for i = 1:length(noise_amplitudes)
        noise_amplitude = noise_amplitudes(i);
        
        % 添加噪声
        noisy_eeg_multichannel = eeg_data_multichannel + noise_amplitude * randn(size(eeg_data_multichannel));
        
        % 应用改进后的CCA降噪
        clean_eeg_multichannel = cca_denoise(noisy_eeg_multichannel);
        
        % 计算评价指标
        selected_channel = 1;
        signal = eeg_data_multichannel(selected_channel, :);
        noisy_signal = noisy_eeg_multichannel(selected_channel, :);
        denoised_signal = clean_eeg_multichannel(selected_channel, :);
        
        % 计算SNR
        noise_signal = noisy_signal - signal;
        signal_power = sum(signal.^2) / length(signal);
        noise_power = sum(noise_signal.^2) / length(noise_signal);
        SNR = 10 * log10(signal_power / noise_power);
        
        % 计算MSE
        MSE = mean((signal - denoised_signal).^2);
        
        % 计算NCC
        ncc = sum(signal .* denoised_signal) / (sqrt(sum(signal.^2)) * sqrt(sum(denoised_signal.^2)));
        
        % 存储结果
        SNR_results(i, c) = SNR;
        MSE_results(i, c) = MSE;
        NCC_results(i, c) = ncc;
    end
end

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
xlabel('Channel Counts');
ylabel('Noise Amplitude');
colorbar;

% 找到最佳通道数（基于最大噪声幅度下的SNR）
[~, best_channel_idx] = max(SNR_results(end, :));
best_channel_number = channel_numbers(best_channel_idx);

% 使用最佳通道数进行最终测试
if best_channel_number == 1
    eeg_data_multichannel_best = EEG_all_epochs(4514, :);
else
    random_indices = randperm(size(EEG_all_epochs, 1), best_channel_number-1);
    eeg_data_multichannel_best = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)];
end

% 添加最大噪声
noisy_eeg_multichannel_best = eeg_data_multichannel_best + 50 * randn(size(eeg_data_multichannel_best));

% 降噪
clean_eeg_multichannel_best = cca_denoise(noisy_eeg_multichannel_best);

% 绘制结果
figure;

% 原始信号图
subplot(3, 1, 1);
plot(eeg_data_multichannel_best(1, :));
title('Original Signal');
ylabel('Amplitude');

% 加噪信号图
subplot(3, 1, 2);
plot(noisy_eeg_multichannel_best(1, :));
title('Noisy Signal (Amplitude 50)');
ylabel('Amplitude');

% 经过CCA降噪后的信号图
subplot(3, 1, 3);
plot(clean_eeg_multichannel_best(1, :));
title('Denoised Signal (CCA)');
ylabel('Amplitude');

% 计算并显示最终结果
signal_best = eeg_data_multichannel_best(1, :);
noisy_signal_best = noisy_eeg_multichannel_best(1, :);
denoised_signal_best = clean_eeg_multichannel_best(1, :);

% 最终SNR
noise_signal_best = noisy_signal_best - signal_best;
signal_power_best = sum(signal_best.^2) / length(signal_best);
noise_power_best = sum(noise_signal_best.^2) / length(noise_signal_best);
SNR_best = 10 * log10(signal_power_best / noise_power_best);

% 最终MSE
MSE_best = mean((signal_best - denoised_signal_best).^2);

% 最终NCC
ncc_best = sum(signal_best .* denoised_signal_best) / (sqrt(sum(signal_best.^2)) * sqrt(sum(denoised_signal_best.^2)));

fprintf('\n=== Final Results ===\n');
fprintf('Best Channel Number: %d\n', best_channel_number);
fprintf('SNR: %.2f dB\n', SNR_best);
fprintf('MSE: %.4f\n', MSE_best);
fprintf('NCC: %.4f\n', ncc_best);
