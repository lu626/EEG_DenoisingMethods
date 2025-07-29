tic;  % >>> 开始记录 Processing Time <<<

%% 加载数据  
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;
inputSize = size(data, 2);
outputSize = inputSize;

% 数据划分
numTrainSamples = 4500;
trainData = data(1:numTrainSamples, :);
testData = data(end-199:end, :);  % 最后200行测试（含第4514行）

% 数据归一化
dataMean = mean(trainData, 'all');
dataStd = std(trainData, 0, 'all');
trainDataNorm = (trainData - dataMean) / dataStd;
testDataNorm = (testData - dataMean) / dataStd;

% 添加噪声（归一化后幅度）
noiseAmplitude = 50 / dataStd;
trainNoisyNorm = trainDataNorm + noiseAmplitude * randn(size(trainDataNorm));
testNoisyNorm  = testDataNorm  + noiseAmplitude * randn(size(testDataNorm));

% 转为 LSTM 输入格式（cell 数组，每个样本为列向量）
train_noisy  = cellfun(@(x) x', num2cell(trainNoisyNorm, 2), 'UniformOutput', false);
train_clean  = cellfun(@(x) x', num2cell(trainDataNorm, 2), 'UniformOutput', false);
test_noisy   = cellfun(@(x) x', num2cell(testNoisyNorm,  2), 'UniformOutput', false);
test_clean   = cellfun(@(x) x', num2cell(testDataNorm,   2), 'UniformOutput', false);

%% 构建简化的 LSTM 网络
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(128, 'OutputMode', 'sequence')
    dropoutLayer(0.1)
    fullyConnectedLayer(outputSize)
    regressionLayer
];

%% 训练选项
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

%% 参数量统计（兼容所有版本）
param_count = 0;
for i = 1:numel(net.Layers)
    if isprop(net.Layers(i), 'Weights') && ~isempty(net.Layers(i).Weights)
        param_count = param_count + numel(net.Layers(i).Weights);
    end
    if isprop(net.Layers(i), 'Bias') && ~isempty(net.Layers(i).Bias)
        param_count = param_count + numel(net.Layers(i).Bias);
    end
end
param_count = param_count / 1e6;  % 单位：M

%% Inference Time 计算（200个样本）
inferTimes = zeros(1, 200);
for i = 1:200
    tStart = tic;
    predict(net, test_noisy{i});
    inferTimes(i) = toc(tStart);
end
avg_infer_time_ms = mean(inferTimes) * 1000;

%% 预测与反归一化
denoisedAll = zeros(size(testData));
for i = 1:length(test_noisy)
    denoised = predict(net, test_noisy{i});
    denoisedAll(i, :) = denoised' * dataStd + dataMean;
end

originalAll = testData;
noisyAll    = testNoisyNorm * dataStd + dataMean;

%% 逐行评估性能指标
SNR_list = zeros(200, 1);
MSE_list = zeros(200, 1);
NCC_list = zeros(200, 1);
for i = 1:200
    clean = originalAll(i, :);
    den   = denoisedAll(i, :);
    SNR_list(i) = 10 * log10(sum(clean.^2) / sum((clean - den).^2));
    MSE_list(i) = mean((clean - den).^2);
    NCC_list(i) = sum(clean .* den) / sqrt(sum(clean.^2) * sum(den.^2));
end

% 输出平均值 ± 标准差
SNR_avg = mean(SNR_list);  SNR_std = std(SNR_list);
MSE_avg = mean(MSE_list);  MSE_std = std(MSE_list);
NCC_avg = mean(NCC_list);  NCC_std = std(NCC_list);

fprintf('\nLSTM [avg over last 200 rows] → SNR = %.2f ± %.2f dB, MSE = %.1f ± %.1f, NCC = %.3f ± %.3f\n', ...
    SNR_avg, SNR_std, MSE_avg, MSE_std, NCC_avg, NCC_std);

%% 绘图（默认绘制第4514行）
figure('Position', [100 100 800 600]);
subplot(3,1,1);
plot(originalAll(end, :)); title('Original Signal (Row 4514)'); ylabel('Amplitude'); grid on;
subplot(3,1,2);
plot(noisyAll(end, :)); title('Noisy Signal'); ylabel('Amplitude'); grid on;
subplot(3,1,3);
plot(denoisedAll(end, :)); title('Denoised Signal'); ylabel('Amplitude'); xlabel('Sample'); grid on;
sgtitle('LSTM-based EEG Denoising Result (Row 4514)');

%% 输出模型统计信息
total_time = toc;
fprintf('Parameter (M): %.2f\n', param_count);
fprintf('Inference Time (ms): %.2f\n', avg_infer_time_ms);
fprintf('Processing Time (s): %.2f\n', total_time);
