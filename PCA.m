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

% 循环不同的通道数量
for c = 1:length(channel_numbers)
num_channels = channel_numbers(c); % 当前通道数量
% 确保包含第4514行
if num_channels == 1
eeg_data_multichannel = EEG_all_epochs(4514, :); % 只选择第4514行
else
random_indices = randperm(size(EEG_all_epochs, 1), num_channels-1); 
eeg_data_multichannel = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)]; % 包含第4514行的多通道数据
end

% 循环不同的噪声幅度
for i = 1:length(noise_amplitudes)
noise_amplitude = noise_amplitudes(i);
% 添加噪声（每个通道单独加噪）
noisy_eeg_multichannel = eeg_data_multichannel + noise_amplitude * randn(size(eeg_data_multichannel));
% PCA 降噪
[coeff, score, latent] = pca(noisy_eeg_multichannel'); % 注意输入为通道 × 时间的转置
num_components_to_keep = find(cumsum(latent) / sum(latent) >= 0.95, 1); % 保留95%的主成分
clean_eeg_multichannel = (score(:, 1:num_components_to_keep) * coeff(:, 1:num_components_to_keep)')';

% 计算 SNR、MSE 和 NCC
% 选择第4514行进行计算
selected_channel = 1; 
% 原始信号（干净的信号）与加噪信号对比
signal = eeg_data_multichannel(selected_channel, :);
noisy_signal = noisy_eeg_multichannel(selected_channel, :);
denoised_signal = clean_eeg_multichannel(selected_channel, :);
% 计算信噪比 (SNR)
noise_signal = noisy_signal - signal;
signal_power = sum(signal.^2) / length(signal);
noise_power = sum(noise_signal.^2) / length(noise_signal);
SNR = 10 * log10(signal_power / noise_power);
% 计算均方误差 (MSE)
MSE = mean((signal - denoised_signal).^2);
% 计算归一化互相关系数 (NCC)
ncc = sum(signal .* denoised_signal) / (sqrt(sum(signal.^2)) * sqrt(sum(denoised_signal.^2)));
% 存储结果
SNR_results(i, c) = SNR;
MSE_results(i, c) = MSE;
NCC_results(i, c) = ncc;
end
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
ylabel('Noise Amplitude');
colorbar;

% ============================
% 选择噪声幅度为50时，最佳通道数
% ============================

% 找到噪声幅度为50时效果最好的通道数（基于SNR）
[~, best_channel_idx] = max(SNR_results(end, :)); % 最后一行对应噪声幅度为50的结果
best_channel_number = channel_numbers(best_channel_idx); % 最佳通道数

% 确保包含第4514行的通道数
if best_channel_number == 1
eeg_data_multichannel_best = EEG_all_epochs(4514, :);
else
random_indices = randperm(size(EEG_all_epochs, 1), best_channel_number-1); 
eeg_data_multichannel_best = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)]; % 包含第4514行的多通道数据
end

% 添加噪声（噪声幅度为50）
noisy_eeg_multichannel_best = eeg_data_multichannel_best + 50 * randn(size(eeg_data_multichannel_best));

% PCA 降噪
[coeff_best, score_best, latent_best] = pca(noisy_eeg_multichannel_best'); 
num_components_to_keep_best = find(cumsum(latent_best) / sum(latent_best) >= 0.95, 1); % 保留95%的主成分
clean_eeg_multichannel_best = (score_best(:, 1:num_components_to_keep_best) * coeff_best(:, 1:num_components_to_keep_best)')';


% 画出原始信号、加噪信号、PCA降噪信号
figure;
subplot(3, 1, 1);
plot(eeg_data_multichannel_best(1, :));
title('Original EEG Signal (Row 4514)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(noisy_eeg_multichannel_best(1, :));
title(['Noisy EEG Signal (Noise Amplitude 50, Channels: ' num2str(best_channel_number) ')']);
ylabel('Amplitude');

subplot(3, 1, 3);
plot(clean_eeg_multichannel_best(1, :));
title('PCA Denoised EEG Signal');
xlabel('Time(s)');
ylabel('Amplitude');


% 计算经过 PCA 降噪后的 SNR, MSE, NCC
signal_best = eeg_data_multichannel_best(1, :);
noisy_signal_best = noisy_eeg_multichannel_best(1, :);
denoised_signal_best = clean_eeg_multichannel_best(1, :);

% 计算信噪比 (SNR)
noise_signal_best = noisy_signal_best - signal_best;
signal_power_best = sum(signal_best.^2) / length(signal_best);
noise_power_best = sum(noise_signal_best.^2) / length(noise_signal_best);
SNR_best = 10 * log10(signal_power_best / noise_power_best);

% 计算均方误差 (MSE)
MSE_best = mean((signal_best - denoised_signal_best).^2);

% 计算归一化互相关系数 (NCC)
ncc_best = sum(signal_best .* denoised_signal_best) / (sqrt(sum(signal_best.^2)) * sqrt(sum(denoised_signal_best.^2)));

% 显示最终结果
fprintf('SNR after PCA: %.4f dB\n', SNR_best);
fprintf('MSE after PCA: %.4f\n', MSE_best);
fprintf('NCC after PCA: %.4f\n', ncc_best);

