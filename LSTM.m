%% 加载数据 
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;
inputSize = size(data, 2);
outputSize = inputSize;

% 数据划分
numTrainSamples = 4500;
trainData = data(1:numTrainSamples, :);
testData = data(end-199:end, :);  %  修改为最后200行测试（含第4514行）

% 数据归一化
dataMean = mean(trainData, 'all');
dataStd = std(trainData, 0, 'all');
trainDataNorm = (trainData - dataMean) / dataStd;
testDataNorm = (testData - dataMean) / dataStd;

% 添加噪声（归一化后幅度）
noiseAmplitude = 50 / dataStd;
trainNoisyNorm = trainDataNorm + noiseAmplitude * randn(size(trainDataNorm));
testNoisyNorm = testDataNorm + noiseAmplitude * randn(size(testDataNorm));

% 转为 LSTM 格式（cell 数组，每个样本为列向量）
train_noisy = cellfun(@(x) x', num2cell(trainNoisyNorm, 2), 'UniformOutput', false);
train_clean = cellfun(@(x) x', num2cell(trainDataNorm, 2), 'UniformOutput', false);
test_noisy = cellfun(@(x) x', num2cell(testNoisyNorm, 2), 'UniformOutput', false);
test_clean = cellfun(@(x) x', num2cell(testDataNorm, 2), 'UniformOutput', false);

%% 构建简化的 LSTM 网络
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(128, 'OutputMode', 'sequence')
    dropoutLayer(0.1)
    fullyConnectedLayer(outputSize)
    regressionLayer
];

%% 优化的训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'GradientThreshold', 1, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

%% 训练网络
net = trainNetwork(train_noisy, train_clean, layers, options);

%% 预测与反归一化
denoisedAll = zeros(size(testData));
for i = 1:length(test_noisy)
    denoised = predict(net, test_noisy{i});
    denoisedAll(i, :) = denoised' * dataStd + dataMean;
end

originalAll = testData;
noisyAll = testNoisyNorm * dataStd + dataMean;

%% 评估指标
SNR = 10 * log10(sum(originalAll.^2, 2) ./ sum((originalAll - denoisedAll).^2, 2));
MSE = mean((originalAll - denoisedAll).^2, 'all');
NCC = sum(originalAll .* denoisedAll, 'all') / sqrt(sum(originalAll.^2, 'all') * sum(denoisedAll.^2, 'all'));

fprintf('\n测试集（最后200行）性能指标：\n');
fprintf('平均 SNR: %.2f dB\n', mean(SNR));
fprintf('平均 MSE: %.6f\n', MSE);
fprintf('平均 NCC: %.4f\n', NCC);

%% 绘图（默认绘制最后一行即第4514行）
figure('Position', [100 100 800 600]);
subplot(3,1,1);
plot(originalAll(end, :)); title('Original Signal (Row 4514)'); ylabel('Amplitude'); grid on;
subplot(3,1,2);
plot(noisyAll(end, :)); title('Noisy Signal'); ylabel('Amplitude'); grid on;
subplot(3,1,3);
plot(denoisedAll(end, :)); title('Denoised Signal'); ylabel('Amplitude'); xlabel('Sample'); grid on;
sgtitle('LSTM-based EEG Denoising Result (Row 4514)');
