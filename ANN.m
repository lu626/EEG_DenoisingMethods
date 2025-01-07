% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取维度
inputSize = size(data, 2);  % 512维的每行EEG信号

% 数据集划分
numSamples = size(data, 1);
trainIdx = 1:round(numSamples * 0.8);
valIdx = (round(numSamples * 0.8) + 1):round(numSamples * 0.95);
testIdx = (round(numSamples * 0.95) + 1):numSamples;

trainData = data(trainIdx, :);
valData = data(valIdx, :);
testData = data(testIdx, :);

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 数据标准化：确保输入信号的范围更一致
trainNoisy = (trainNoisy - mean(trainNoisy, 1)) ./ std(trainNoisy, 0, 1);
valNoisy = (valNoisy - mean(valNoisy, 1)) ./ std(valNoisy, 0, 1);
testNoisy = (testNoisy - mean(testNoisy, 1)) ./ std(testNoisy, 0, 1);

% 数据重塑为 [样本数, 特征数] 格式
train_noisy = trainNoisy;  % [样本数, 特征数]
train_clean = trainData;

val_noisy = valNoisy;  % [样本数, 特征数]
val_clean = valData;

test_noisy = testNoisy;  % [样本数, 特征数]
test_clean = testData;

% ========================== Adaline模型 =========================

% 初始化Adaline参数
learningRate = 0.05;  % 学习率
epochs = 400;          % 训练轮数
inputSize = size(train_noisy, 2);  % 输入维度，512个特征

% 权重初始化
weights = zeros(inputSize, 1);
bias = 0;

% 训练Adaline模型
for epoch = 1:epochs
    % 计算预测值
    y_pred = train_noisy * weights + bias;
    
    % 计算误差
    error = train_clean - y_pred;
    
    % 更新权重和偏置
    weights = weights + learningRate * (train_noisy' * error) / size(train_noisy, 1);
    bias = bias + learningRate * mean(error);
    
    % 计算训练误差
    MSE = mean(error.^2);
    
    % 每10轮输出一次训练误差
    if mod(epoch, 10) == 0
        fprintf('Epoch: %d, MSE: %.5f\n', epoch, MSE);
    end
end

% ========================== 应用Adaline模型 =========================

% 使用训练好的Adaline模型对第4514行数据进行降噪
test_noisy_4514 = test_noisy(end, :);  % 使用最后一行数据
test_clean_4514 = test_clean(end, :);  % 使用最后一行干净数据

% 预测降噪信号
denoisedSignal_4514 = test_noisy_4514 * weights + bias;

% 绘图
figure;

subplot(3, 1, 1);
plot(test_clean_4514);
title('Original Signal');
ylabel('EEG Signal');

subplot(3, 1, 2);
plot(test_noisy_4514);
title('Noisy Signal');
ylabel('EEG Signal');

subplot(3, 1, 3);
plot(denoisedSignal_4514);
title('Denoised Signal (Adaline)');
ylabel('EEG Signal');

% 计算性能指标
SNR_4514 = 10 * log10(sum(test_clean_4514.^2) / sum((test_clean_4514 - denoisedSignal_4514).^2));
MSE_4514 = mean((test_clean_4514 - denoisedSignal_4514).^2);
NCC_4514 = sum(test_clean_4514 .* denoisedSignal_4514) / sqrt(sum(test_clean_4514.^2) * sum(denoisedSignal_4514.^2));

% 输出性能指标
fprintf('SNR (4514th): %.5f\n', SNR_4514);
fprintf('MSE (4514th): %.5f\n', MSE_4514);
fprintf('NCC (4514th): %.5f\n', NCC_4514);
