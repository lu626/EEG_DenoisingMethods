% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取维度
inputSize = size(data, 2);
numChannels = 1;

% 数据集划分
numSamples = size(data, 1);
trainIdx = 1:round(numSamples * 0.8);
valIdx = (round(numSamples * 0.8) + 1):round(numSamples * 0.95);
testIdx = (round(numSamples * 0.95) + 1):numSamples;

trainData = data(trainIdx, :);
valData = data(valIdx, :);
testData = data(testIdx, :);

% 选择第4514行数据进行测试
testRowIdx = 4514;
testData = data(testRowIdx, :);  % 只选择第4514行

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 重构数据格式为 [height width channels samples]
train_noisy = permute(reshape(trainNoisy, [size(trainNoisy, 1), 1, inputSize, 1]), [3 2 4 1]);
train_clean = permute(reshape(trainData, [size(trainData, 1), 1, inputSize, 1]), [3 2 4 1]);

val_noisy = permute(reshape(valNoisy, [size(valNoisy, 1), 1, inputSize, 1]), [3 2 4 1]);
val_clean = permute(reshape(valData, [size(valData, 1), 1, inputSize, 1]), [3 2 4 1]);

test_noisy = permute(reshape(testNoisy, [size(testNoisy, 1), 1, inputSize, 1]), [3 2 4 1]);
test_clean = permute(reshape(testData, [size(testData, 1), 1, inputSize, 1]), [3 2 4 1]);

% 定义 U-Net 网络结构
layers = [
    imageInputLayer([inputSize 1 1])

    % Encoder
    convolution2dLayer([3 1], 64, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer([2 1], 'Stride', [2 1])

    convolution2dLayer([3 1], 128, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer([2 1], 'Stride', [2 1])

    % Bottleneck
    convolution2dLayer([3 1], 256, 'Padding', 'same')
    reluLayer

    % Decoder
    transposedConv2dLayer([2 1], 128, 'Stride', [2 1], 'Cropping', 'same')
    reluLayer

    transposedConv2dLayer([2 1], 64, 'Stride', [2 1], 'Cropping', 'same')
    reluLayer

    % Output layer
    convolution2dLayer([3 1], 1, 'Padding', 'same')
    regressionLayer
];

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...  % 增加训练轮数
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.0005, ...  % 降低学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'ValidationData', {val_noisy, val_clean}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 训练网络
net = trainNetwork(train_noisy, train_clean, layers, options);

% 使用网络进行预测
denoisedSignal = predict(net, test_noisy);

% 将数据转换回原始维度用于绘图
denoisedSignal = permute(denoisedSignal, [4 3 2 1]);
originalSignal = permute(test_clean, [4 3 2 1]);
noisySignal = permute(test_noisy, [4 3 2 1]);

% 绘图
figure;
subplot(3, 1, 1);
plot(squeeze(originalSignal(1, :)));
title('Original Signal');
subplot(3, 1, 2);
plot(squeeze(noisySignal(1, :)));
title('Noisy Signal');
subplot(3, 1, 3);
plot(squeeze(denoisedSignal(1, :)));
title('Denoised Signal');

% 计算性能指标
SNR = 10 * log10(sum(originalSignal.^2, 2) ./ sum((originalSignal - denoisedSignal).^2, 2));
MSE = mean((originalSignal - denoisedSignal).^2, 'all');
NCC = sum(originalSignal .* denoisedSignal, 'all') / sqrt(sum(originalSignal.^2, 'all') * sum(denoisedSignal.^2, 'all'));

% 输出性能指标
fprintf('SNR: %.5f\n', mean(SNR));
fprintf('MSE: %.5f\n', MSE);
fprintf('NCC: %.5f\n', NCC);
