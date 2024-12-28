% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;
inputSize = size(data, 2);
outputSize = size(data, 2);

% 只选择500行用于训练
numTrainSamples = 4500;
trainIdx = 1:numTrainSamples;
trainData = data(trainIdx, :);

% 选择第4514行作为测试数据
testData = data(4514, :);

% 数据归一化
dataMean = mean(trainData, 'all');
dataStd = std(trainData, [], 'all');
trainData = (trainData - dataMean) / dataStd;
testData = (testData - dataMean) / dataStd;

% 添加噪声
noiseAmplitude = 50/dataStd; % 归一化后的噪声幅度
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 转换为LSTM格式
train_noisy = cellfun(@(x) x', num2cell(trainNoisy, 2), 'UniformOutput', false);
train_clean = cellfun(@(x) x', num2cell(trainData, 2), 'UniformOutput', false);
test_noisy = {testData'}; % 单个测试样本
test_clean = {testData'};

% 简化的LSTM网络结构
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(128, 'OutputMode', 'sequence')
    dropoutLayer(0.1)
    fullyConnectedLayer(outputSize)
    regressionLayer
];

% 优化的训练参数
options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'GradientThreshold', 1, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'gpu');

% 训练网络
net = trainNetwork(train_noisy, train_clean, layers, options);

% 预测与反归一化
denoisedSignal = predict(net, test_noisy);
denoisedSignal = cell2mat(denoisedSignal)' * dataStd + dataMean;
originalSignal = testData * dataStd + dataMean;
noisySignal = cell2mat(test_noisy)' * dataStd + dataMean;

% 绘图
figure;
subplot(3,1,1);
plot(originalSignal);
title('Original Signal');
subplot(3,1,2);
plot(noisySignal);
title('Noisy Signal');
subplot(3,1,3);
plot(denoisedSignal);
title('Denoised Signal');

% 计算性能指标
SNR = 10 * log10(sum(originalSignal.^2) / sum((originalSignal - denoisedSignal).^2));
MSE = mean((originalSignal - denoisedSignal).^2);
NCC = sum(originalSignal .* denoisedSignal) / sqrt(sum(originalSignal.^2) * sum(denoisedSignal.^2));

fprintf('SNR: %.5f\n', SNR);
fprintf('MSE: %.5f\n', MSE);
fprintf('NCC: %.5f\n', NCC);