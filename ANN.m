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

% 重要：数据重塑为 [samples, features] 格式
train_noisy = trainNoisy;  % [样本数, 特征数]
train_clean = trainData;

val_noisy = valNoisy;  % [样本数, 特征数]
val_clean = valData;

test_noisy = testNoisy;  % [样本数, 特征数]
test_clean = testData;

% 定义网络结构 (简单网络)
layers = [
    featureInputLayer(inputSize)  % 输入层的大小是512维信号
    
    fullyConnectedLayer(64)       % 第一个全连接层，较少神经元数量
    reluLayer
    
    fullyConnectedLayer(32)       % 第二个全连接层
    reluLayer
    
    fullyConnectedLayer(inputSize) % 输出层的神经元数量与输入数据的维度一致
    regressionLayer                % 回归层
];

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...  % 使用100轮
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...  % 学习率不变
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'ValidationData', {val_noisy, val_clean}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 训练网络
net = trainNetwork(train_noisy, train_clean, layers, options);

% 只对第4514行数据进行降噪 (修改为访问最后一行数据)
test_noisy_4514 = test_noisy(end, :);  % 使用最后一行数据
test_clean_4514 = test_clean(end, :);  % 使用最后一行干净数据

% 使用网络进行预测
denoisedSignal_4514 = predict(net, test_noisy_4514);

% 绘图
figure;

subplot(3, 1, 3);
plot(denoisedSignal_4514);
ylabel('ANN');

% 计算性能指标 (针对第4514行)
SNR_4514 = 10 * log10(sum(test_clean_4514.^2) / sum((test_clean_4514 - denoisedSignal_4514).^2));
MSE_4514 = mean((test_clean_4514 - denoisedSignal_4514).^2);
NCC_4514 = sum(test_clean_4514 .* denoisedSignal_4514) / sqrt(sum(test_clean_4514.^2) * sum(denoisedSignal_4514.^2));

% 输出性能指标
fprintf('SNR (4514th): %.5f\n', SNR_4514);
fprintf('MSE (4514th): %.5f\n', MSE_4514);
fprintf('NCC (4514th): %.5f\n', NCC_4514);
