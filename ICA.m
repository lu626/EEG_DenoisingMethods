%% 启动计时
tic;  % 开始总处理时间计时

%% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');

%% 参数设置
noise_amplitude = 50;             % 噪声幅度
best_channel_number = 300;        % 可手动设置或来自热图分析结果
target_idx = 4514;                % 目标 EEG 行

%% 构造数据（确保包含目标行）
if best_channel_number == 1
    eeg_data_multichannel = EEG_all_epochs(target_idx, :);
else
    available_indices = setdiff(1:size(EEG_all_epochs,1), target_idx);
    rand_idx = randperm(length(available_indices), best_channel_number - 1);
    eeg_data_multichannel = [EEG_all_epochs(target_idx, :); EEG_all_epochs(available_indices(rand_idx), :)];
end

%% 添加噪声
rng(0);  % 固定随机种子
noise = noise_amplitude * randn(size(eeg_data_multichannel));
noisy_data = eeg_data_multichannel + noise;

%% ICA 降噪
if best_channel_number > 1
    [S, A, W] = fastica(noisy_data', ...
        'approach', 'symm', ...
        'g', 'tanh', ...
        'maxNumIterations', 500, ...
        'epsilon', 1e-3);
    clean_data = (A * S)';  % 正确的信号重构方式
else
    clean_data = noisy_data;
end

%% 提取目标信号
original_signal = eeg_data_multichannel(1, :);
noisy_signal = noisy_data(1, :);
denoised_signal = clean_data(1, :);

%% 计算指标
signal_power = sum(original_signal.^2) / length(original_signal);
noise_power = sum((noisy_signal - original_signal).^2) / length(original_signal);
SNR = 10 * log10(signal_power / noise_power);
MSE = mean((original_signal - denoised_signal).^2);
NCC = sum((original_signal - mean(original_signal)) .* (denoised_signal - mean(denoised_signal))) / ...
    sqrt(sum((original_signal - mean(original_signal)).^2) * sum((denoised_signal - mean(denoised_signal)).^2));

%% 停止计时
processing_time = toc;

%% 输出指标
fprintf('\nFinal Results (ICA with %d channels, Amp=%d):\n', best_channel_number, noise_amplitude);
fprintf('SNR: %.4f dB\n', SNR);
fprintf('MSE: %.4f\n', MSE);
fprintf('NCC: %.4f\n', NCC);

% 输出参数量（ICA为非深度模型，无可训练参数）
fprintf('ICA Parameter Count: N/A (non-trainable method)\n');
fprintf('ICA Processing Time: %.4f seconds\n', processing_time);

%% 绘图
figure('Position', [100, 100, 800, 600]);

subplot(3,1,1);
plot(original_signal);
title('Original Signal (Row 4514)');
ylabel('Amplitude');

subplot(3,1,2);
plot(noisy_signal);
title(sprintf('Noisy Signal (Amp=%d, Channels=%d)', noise_amplitude, best_channel_number));
ylabel('Amplitude');

subplot(3,1,3);
plot(denoised_signal);
title('ICA Denoised Signal');
xlabel('Time Points');
ylabel('Amplitude');
