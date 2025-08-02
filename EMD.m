% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
tic;  % 开始统计 processing time

% 定义参数
noise_amplitudes = 5:5:50;  % 噪声幅度范围
channel_numbers = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]; % 选择不同的通道数
SNR_results = zeros(length(noise_amplitudes), length(channel_numbers)); % 存储SNR结果
MSE_results = zeros(length(noise_amplitudes), length(channel_numbers)); % 存储MSE结果
NCC_results = zeros(length(noise_amplitudes), length(channel_numbers)); % 存储NCC结果

% 主循环
for c = 1:length(channel_numbers)
    num_channels = channel_numbers(c);
    if num_channels == 1
        eeg_data_multichannel = EEG_all_epochs(4514, :);  % 只选第4514行数据作为单通道
    else
        random_indices = randperm(size(EEG_all_epochs, 1), num_channels-1);
        eeg_data_multichannel = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)];
    end
    
    for i = 1:length(noise_amplitudes)
        noise_amplitude = noise_amplitudes(i);
        noisy_eeg_multichannel = eeg_data_multichannel + noise_amplitude * randn(size(eeg_data_multichannel));
        
        % EMD降噪
        clean_eeg_multichannel = noisy_eeg_multichannel;
        for ch = 1:num_channels
            % 调整EMD参数
            [imfs, residual] = emd(noisy_eeg_multichannel(ch, :), ...
                'MaxNumIMF', 6, ...           % 限制IMF数量
                'MaxNumExtrema', 4, ...       % 设置较小的极值数
                'SiftMaxIterations', 200, ... % 增加最大迭代次数
                'SiftRelativeTolerance', 0.01); % 更严格的收敛容忍度
            
            if size(imfs, 2) == length(noisy_eeg_multichannel(ch, :))
                % 改进的IMF选择策略
                num_imfs = size(imfs, 1);
                
                % 计算每个IMF的能量
                imf_energies = sum(imfs.^2, 2);
                total_energy = sum(imf_energies);
                energy_ratio = imf_energies / total_energy;
                
                % 基于噪声幅度的自适应阈值
                threshold = 0.2 * exp(-noise_amplitude / 30);
                
                % 选择能量比例大于阈值的IMF
                selected_imfs = energy_ratio > threshold;
                imfs_denoised = imfs(selected_imfs, :);
                
                if isempty(imfs_denoised)
                    imfs_denoised = imfs(1:ceil(num_imfs / 2), :);  % 如果没有选中的IMF，则选择前半部分
                end
                
                clean_eeg_multichannel(ch, :) = sum(imfs_denoised, 1) + residual;
            else
                clean_eeg_multichannel(ch, :) = noisy_eeg_multichannel(ch, :);
            end
        end
        
        % 评估指标计算（与论文公式一致）
        selected_channel = 1;
        signal = eeg_data_multichannel(selected_channel, :);          % X
        denoised_signal = clean_eeg_multichannel(selected_channel, :);% Z
        
        % SNR: 10*log10( sum(X^2) / sum((X-Z)^2) )
        SNR = 10 * log10( sum(signal.^2) / sum( (signal - denoised_signal).^2 ) );
        
        % MSE: mean( (X - Z).^2 )
        MSE = mean((signal - denoised_signal).^2);
        
        % NCC: 去均值后的归一化相关
        xs = signal - mean(signal);
        zs = denoised_signal - mean(denoised_signal);
        ncc = sum(xs .* zs) / sqrt(sum(xs.^2) * sum(zs.^2));
        
        % 存储评估结果
        SNR_results(i, c) = SNR;
        MSE_results(i, c) = MSE;
        NCC_results(i, c) = ncc;
    end
end

figure;

% 绘制 SNR 热图
subplot(3, 1, 1);
h1 = heatmap(channel_numbers, noise_amplitudes, SNR_results);
title('SNR Heatmap');
h1.XLabel = 'Channel Counts (EMD)';  
h1.YLabel = 'Noise Amplitude';
colorbar;

% 绘制 MSE 热图
subplot(3, 1, 2);
h2 = heatmap(channel_numbers, noise_amplitudes, MSE_results);
title('MSE Heatmap');
h2.XLabel = 'Channel Counts (EMD)'; 
h2.YLabel = 'Noise Amplitude';
colorbar;

% 绘制 NCC 热图
subplot(3, 1, 3);
h3 = heatmap(channel_numbers, noise_amplitudes, NCC_results);
title('NCC Heatmap');
h3.XLabel = 'Channel Counts (EMD)';   
h3.YLabel = 'Noise Amplitude';
colorbar;

% 最佳通道分析（以 A=50 的 SNR 为准）
[~, best_channel_idx] = max(SNR_results(end, :));
best_channel_number = channel_numbers(best_channel_idx);

if best_channel_number == 1
    eeg_data_multichannel_best = EEG_all_epochs(4514, :);
else
    random_indices = randperm(size(EEG_all_epochs, 1), best_channel_number-1);
    eeg_data_multichannel_best = [EEG_all_epochs(4514, :); EEG_all_epochs(random_indices, :)];
end

noisy_eeg_multichannel_best = eeg_data_multichannel_best + 50 * randn(size(eeg_data_multichannel_best));

% EMD降噪最佳通道
clean_eeg_multichannel_best = noisy_eeg_multichannel_best;
for ch = 1:best_channel_number
    [imfs_best, residual_best] = emd(noisy_eeg_multichannel_best(ch, :), ...
        'MaxNumIMF', 6, 'MaxNumExtrema', 4, 'SiftMaxIterations', 200, 'SiftRelativeTolerance', 0.01);
    
    if size(imfs_best, 2) == length(noisy_eeg_multichannel_best(ch, :))
        num_imfs_best = size(imfs_best, 1);
        
        % 计算IMF能量
        imf_energies = sum(imfs_best.^2, 2);
        total_energy = sum(imf_energies);
        energy_ratio = imf_energies / total_energy;
        
        % 自适应阈值
        threshold = 0.2 * exp(-50 / 30);
        selected_imfs = energy_ratio > threshold;
        imfs_denoised_best = imfs_best(selected_imfs, :);
        
        if isempty(imfs_denoised_best)
            imfs_denoised_best = imfs_best(1:ceil(num_imfs_best / 2), :);
        end
        
        clean_eeg_multichannel_best(ch, :) = sum(imfs_denoised_best, 1) + residual_best;
    else
        clean_eeg_multichannel_best(ch, :) = noisy_eeg_multichannel_best(ch, :);
    end
end

% 结果可视化
figure;
subplot(3, 1, 3);
plot(clean_eeg_multichannel_best(1, :));
ylabel('EMD');

% 最终评估指标（单次：4514 @ A=50, 最佳通道数）
signal_best = eeg_data_multichannel_best(1, :);
denoised_signal_best = clean_eeg_multichannel_best(1, :);

SNR_best = 10 * log10( sum(signal_best.^2) / sum((signal_best - denoised_signal_best).^2) );
MSE_best = mean((signal_best - denoised_signal_best).^2);
xs_b = signal_best - mean(signal_best);
zs_b = denoised_signal_best - mean(denoised_signal_best);
ncc_best = sum(xs_b .* zs_b) / sqrt(sum(xs_b.^2) * sum(zs_b.^2));

fprintf('SNR after EMD: %.4f dB\n', SNR_best);
fprintf('MSE after EMD: %.4f\n', MSE_best);
fprintf('NCC after EMD: %.4f\n', ncc_best);

% 输出参数量与运行时间
param_count = 0;  % EMD 为传统方法，无可训练参数
processing_time = toc;

fprintf('Parameter Count: %d\n', param_count);
fprintf('Processing Time: %.4f seconds\n', processing_time);
