%% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取维度
inputSize = size(data, 2);


% 数据划分
numSamples = size(data, 1);
trainIdx = 1:round(numSamples * 0.8);
valIdx = (round(numSamples * 0.8) + 1):round(numSamples * 0.95);
testEvalIdx = (numSamples - 199):numSamples;  % ✅ 改为倒数200行用于评估
testPlotIdx = 4514;  % 第4514行用于绘图

trainData = data(trainIdx, :);
valData = data(valIdx, :);
testEvalData = data(testEvalIdx, :);
testPlotData = data(testPlotIdx, :);

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testEvalNoisy = testEvalData + noiseAmplitude * randn(size(testEvalData));
testPlotNoisy = testPlotData + noiseAmplitude * randn(size(testPlotData));

% 重塑数据为 [W, H, C, N]
reshape4D = @(x) permute(reshape(x, [size(x,1), 1, inputSize, 1]), [3 2 4 1]);
train_noisy = reshape4D(trainNoisy);
train_clean = reshape4D(trainData);
val_noisy = reshape4D(valNoisy);
val_clean = reshape4D(valData);
testEval_noisy = reshape4D(testEvalNoisy);
testEval_clean = reshape4D(testEvalData);
testPlot_noisy = reshape4D(testPlotNoisy);
testPlot_clean = reshape4D(testPlotData);

%% 定义网络结构
layers = [
    imageInputLayer([inputSize 1 1])

    convolution2dLayer([7 1], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([7 1], 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([7 1], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([5 1], 1, 'Padding', 'same')
    regressionLayer
];

%% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.0005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 50, ...
    'ValidationData', {val_noisy, val_clean}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% 训练网络
net = trainNetwork(train_noisy, train_clean, layers, options);

%% 使用网络预测 - 4514行用于绘图
denoisedPlot = predict(net, testPlot_noisy);
oriPlot = permute(testPlot_clean, [4 3 2 1]);
noisyPlot = permute(testPlot_noisy, [4 3 2 1]);
denoisedPlot = permute(denoisedPlot, [4 3 2 1]);

figure;
subplot(3, 1, 1); plot(squeeze(oriPlot(1, :))); title('Original Signal');
subplot(3, 1, 2); plot(squeeze(noisyPlot(1, :))); title('Noisy Signal');
subplot(3, 1, 3); plot(squeeze(denoisedPlot(1, :))); title('Denoised Signal');

%% 对倒数20行计算平均 SNR/MSE/NCC
denoisedEval = predict(net, testEval_noisy);
denoisedEval = permute(denoisedEval, [4 3 2 1]);
oriEval = permute(testEval_clean, [4 3 2 1]);

% 初始化
SNR_list = zeros(size(oriEval,1),1);
MSE_list = zeros(size(oriEval,1),1);
NCC_list = zeros(size(oriEval,1),1);

for i = 1:size(oriEval,1)
    sig = oriEval(i,:);
    den = denoisedEval(i,:);
    SNR_list(i) = 10 * log10(sum(sig.^2) / sum((sig - den).^2));
    MSE_list(i) = mean((sig - den).^2);
    NCC_list(i) = sum(sig .* den) / sqrt(sum(sig.^2) * sum(den.^2));
end

% 输出平均指标
fprintf('\n【Test Set: Last 200 Rows Including Row 4514】\n');  % 这里也同步修改文本
fprintf('Avg SNR: %.4f dB\n', mean(SNR_list));
fprintf('Avg MSE: %.4f\n', mean(MSE_list));
fprintf('Avg NCC: %.4f\n', mean(NCC_list));

