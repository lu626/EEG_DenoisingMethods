% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
% 选择第4514行的数据（假设EEG数据集是一个4514×512的矩阵）
original_signal = EEG_all_epochs(4514, :);

% 设置噪声幅度为50
noise_amplitude = 50;

% 为信号添加噪声
noisy_signal = original_signal + noise_amplitude * randn(size(original_signal));

% 多维卡尔曼滤波器设计
% 初始化卡尔曼滤波器的参数
process_noise = 1e-3;  % 增加过程噪声（适度增加信号的动态变化）
measurement_noise = 5e-2;  % 测量噪声（稍微降低，使滤波器更依赖观测信号）
initial_estimate = [0; 0; 0];  % 初始估计值（位置、速度、加速度）
initial_error_cov = eye(3) * 10;  % 初始估计误差协方差（稍大值，加速收敛）

% 状态转移矩阵：我们假设信号的状态由位置、速度和加速度组成
F = [1, 1, 0.5; 0, 1, 1; 0, 0, 1];  % 状态转移矩阵（位置、速度、加速度）

% 观测矩阵：我们假设我们只观测位置
H = [1, 0, 0];  % 观测矩阵（只观测位置）

% 初始化变量
n = length(noisy_signal);  % 信号的长度
estimated_signal = zeros(1, n);  % 存储滤波后的信号
P = initial_error_cov;  % 初始估计的协方差矩阵
x_hat = initial_estimate;  % 初始估计的值
estimation_error = zeros(1, n);  % 存储每个时刻的估计误差

% 卡尔曼滤波迭代
for k = 1:n
    % 预测阶段
    x_hat_predict = F * x_hat;  % 预测的状态
    P_predict = F * P * F' + process_noise * eye(3);  % 预测的误差协方差

    % 更新阶段
    K = P_predict * H' / (H * P_predict * H' + measurement_noise);  % 卡尔曼增益
    x_hat = x_hat_predict + K * (noisy_signal(k) - H * x_hat_predict);  % 更新估计
    P = (eye(3) - K * H) * P_predict;  % 更新误差协方差
    
    % 在误差协方差更新时加入一个正则化项
    P = P + 1e-5 * eye(3);  % 防止误差协方差矩阵过小或奇异

    % 存储滤波后的信号（我们关心的是位置，即x_hat(1)）
    estimated_signal(k) = x_hat(1);

    % 计算估计误差
    estimation_error(k) = noisy_signal(k) - x_hat(1);
end

% 计算 SNR, MSE 和 NCC

% 计算 SNR
signal_power = sum(original_signal.^2);
noise_power = sum((original_signal - noisy_signal).^2);
SNR = 10 * log10(signal_power / noise_power);

% 计算 MSE
MSE = mean((original_signal - estimated_signal).^2);

% 计算 NCC
ncc = sum((original_signal - mean(original_signal)) .* (estimated_signal - mean(estimated_signal))) / ...
      sqrt(sum((original_signal - mean(original_signal)).^2) * sum((estimated_signal - mean(estimated_signal)).^2));

% 输出 SNR, MSE 和 NCC
fprintf('SNR: %.4f dB\n', SNR);
fprintf('MSE: %.4f\n', MSE);
fprintf('NCC: %.4f\n', ncc);

% 绘制图形：原始信号、加噪信号、卡尔曼滤波后的信号、估计误差
figure;

% 原始信号
subplot(3, 1, 1);
plot(original_signal);
title('Original EEG Signal (Row 4514)');


% 加噪信号
subplot(3, 1, 2);
plot(noisy_signal);
ylabel('NS');


% 多维卡尔曼滤波后的信号
subplot(3, 1, 3);
plot(estimated_signal);
ylabel('Filtering');


