% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');

% 定义噪声幅度和通道数量
noise_amplitudes = 5:5:50;  % 噪声幅度从5到50，步长为5
channel_numbers = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];  % 通道数

% 存储结果
SNR_results = zeros(length(noise_amplitudes), length(channel_numbers));
MSE_results = zeros(length(noise_amplitudes), length(channel_numbers));
NCC_results = zeros(length(noise_amplitudes), length(channel_numbers));

% 优化ICA参数
max_iterations = 100;  % 限制最大迭代次数
epsilon = 1e-3;        % 收敛阈值

% 循环不同通道数量
for c = 1:length(channel_numbers)
    num_channels = channel_numbers(c);
    
    % 确保包含第4514行
    if num_channels == 1
        eeg_data_multichannel = EEG_all_epochs(4514, :);  % 只有1个通道时，选择第4514行
    else
        % 随机选择其他通道，但确保包含第4514行
        available_indices = setdiff(1:size(EEG_all_epochs,1), 4514);
        random_indices = available_indices(randperm(length(available_indices), num_channels-1));
        eeg_data_multichannel = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)];
    end
    
    % 循环不同的噪声幅度
    for i = 1:length(noise_amplitudes)
        noise_amplitude = noise_amplitudes(i);
        
        % 添加噪声
        noise = noise_amplitude * randn(size(eeg_data_multichannel));
        noisy_eeg_multichannel = eeg_data_multichannel + noise;
        
        % ICA降噪
        try
            if num_channels > 1
                % 使用优化后的ICA参数
                [S, A, W] = fastica(noisy_eeg_multichannel', ...
                    'approach', 'symm', ...
                    'g', 'tanh', ...
                    'maxNumIterations', max_iterations, ...
                    'epsilon', epsilon, ...
                    'verbose', 'off');
                
                % 重建信号
                clean_eeg_multichannel = (W \ S)'; 
            else
                % 单通道情况
                clean_eeg_multichannel = noisy_eeg_multichannel;
            end
            
            % 获取第4514行对应的信号
            original_signal = eeg_data_multichannel(1, :);
            noisy_signal = noisy_eeg_multichannel(1, :);
            denoised_signal = clean_eeg_multichannel(1, :);
            
            % 计算指标
            noise_signal = noisy_signal - original_signal;
            signal_power = sum(original_signal.^2) / length(original_signal);
            noise_power = sum(noise_signal.^2) / length(noise_signal);
            SNR = 10 * log10(signal_power / noise_power);
            
            MSE = mean((original_signal - denoised_signal).^2);
            NCC = sum(original_signal .* denoised_signal) / ...
                (sqrt(sum(original_signal.^2)) * sqrt(sum(denoised_signal.^2)));
            
            % 存储结果
            SNR_results(i, c) = SNR;
            MSE_results(i, c) = MSE;
            NCC_results(i, c) = NCC;
            
        catch ME
            fprintf('Error at channels=%d, noise=%d: %s\n', num_channels, noise_amplitude, ME.message);
            % 出错时设置为NaN而不是继续
            SNR_results(i, c) = NaN;
            MSE_results(i, c) = NaN;
            NCC_results(i, c) = NaN;
        end
    end
end

% 创建一个新的图形并缩小窗口
figure('Position', [200, 200, 600, 600]);  % 修改图形窗口大小

% 绘制 SNR 热图
subplot(3, 1, 1); % 第一行
h1 = heatmap(channel_numbers, noise_amplitudes, SNR_results);
h1.Title = 'SNR Heatmap';
h1.YLabel = 'Noise Amplitude';
h1.CellLabelColor = 'none';  % 禁用数字显示
colorbar;

% 绘制 MSE 热图
subplot(3, 1, 2); % 第二行
h2 = heatmap(channel_numbers, noise_amplitudes, MSE_results);
h2.Title = 'MSE Heatmap';
h2.YLabel = 'Noise Amplitude';
h2.CellLabelColor = 'none';  % 禁用数字显示
colorbar;

% 绘制 NCC 热图
subplot(3, 1, 3); % 第三行
h3 = heatmap(channel_numbers, noise_amplitudes, NCC_results);
h3.Title = 'NCC Heatmap';
h3.YLabel = 'Noise Amplitude';
h3.CellLabelColor = 'none';  % 禁用数字显示
colorbar;

% 找到噪声幅度为50时的最佳通道数
[~, best_channel_idx] = max(SNR_results(end, :));
best_channel_number = channel_numbers(best_channel_idx);
fprintf('Best channel number when noise amplitude is 50: %d\n', best_channel_number);

% 使用最佳通道数进行最终测试
if best_channel_number == 1
    eeg_data_multichannel_best = EEG_all_epochs(4514, :);
else
    available_indices = setdiff(1:size(EEG_all_epochs,1), 4514);
    random_indices = available_indices(randperm(length(available_indices), best_channel_number-1));
    eeg_data_multichannel_best = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)];
end

% 添加噪声进行最终测试
noise_amplitude = 50;
noise_best = noise_amplitude * randn(size(eeg_data_multichannel_best));
noisy_eeg_multichannel_best = eeg_data_multichannel_best + noise_best;

% 最终ICA降噪
if best_channel_number > 1
    [S_best, ~, W_best] = fastica(noisy_eeg_multichannel_best', ...
        'approach', 'symm', ...
        'g', 'tanh', ...
        'maxNumIterations', max_iterations, ...
        'epsilon', epsilon, ...
        'verbose', 'off');
    clean_eeg_multichannel_best = (W_best \ S_best)'; 
else
    clean_eeg_multichannel_best = noisy_eeg_multichannel_best;
end

% 绘制最终结果
figure('Position', [100, 100, 800, 600]);

subplot(3,1,1);
plot(eeg_data_multichannel_best(1,:));
title('Original Signal (Row 4514)');
ylabel('Amplitude');

subplot(3,1,2);
plot(noisy_eeg_multichannel_best(1,:));
title(['Noisy Signal (Amplitude: 50, Channels: ' num2str(best_channel_number) ')']);
ylabel('Amplitude');

subplot(3,1,3);
plot(clean_eeg_multichannel_best(1,:));
title('ICA Denoised Signal');
xlabel('Time Points');
ylabel('Amplitude');

% 计算并显示最终结果
original_signal_best = eeg_data_multichannel_best(1, :);
noisy_signal_best = noisy_eeg_multichannel_best(1, :);
denoised_signal_best = clean_eeg_multichannel_best(1, :);

noise_signal_best = noisy_signal_best - original_signal_best;
signal_power_best = sum(original_signal_best.^2) / length(original_signal_best);
noise_power_best = sum(noise_signal_best.^2) / length(noise_signal_best);
SNR_best = 10 * log10(signal_power_best / noise_power_best);

MSE_best = mean((original_signal_best - denoised_signal_best).^2);
NCC_best = sum(original_signal_best .* denoised_signal_best) / ...
    (sqrt(sum(original_signal_best.^2)) * sqrt(sum(denoised_signal_best.^2)));

fprintf('\nFinal Results with Best Channel Configuration:\n');
fprintf('SNR: %.4f dB\n', SNR_best);
fprintf('MSE: %.4f\n', MSE_best);
fprintf('NCC: %.4f\n', NCC_best);
