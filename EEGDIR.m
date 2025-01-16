% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取维度
[numSamples, signalLength] = size(data); % 4514 x 512

% 数据集划分
trainIdx = 1:round(numSamples * 0.8);
valIdx = (round(numSamples * 0.8) + 1):round(numSamples * 0.95);
testIdx = numSamples;  % 只取最后一行作为测试集

trainData = data(trainIdx, :);
valData = data(valIdx, :);
testData = data(testIdx, :);  % 测试集仅为最后一行数据

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 将数据重塑为 [height width channels samples] 格式
train_noisy = reshape(trainNoisy', [signalLength, 1, 1, size(trainNoisy, 1)]);
train_clean = reshape(trainData', [signalLength, 1, 1, size(trainData, 1)]);
val_noisy = reshape(valNoisy', [signalLength, 1, 1, size(valNoisy, 1)]);
val_clean = reshape(valData', [signalLength, 1, 1, size(valData, 1)]);
test_noisy = reshape(testNoisy', [signalLength, 1, 1, size(testNoisy, 1)]);
test_clean = reshape(testData', [signalLength, 1, 1, size(testData, 1)]);

% 定义网络结构
layers = [
    imageInputLayer([signalLength 1 1], 'Name', 'input')
    
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    convolution2dLayer([3 1], 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    
    convolution2dLayer([3 1], 1, 'Padding', 'same', 'Name', 'conv6')
    regressionLayer('Name', 'output')
];

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'ValidationData', {val_noisy, val_clean}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 5, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

% 训练网络
net = trainNetwork(train_noisy, train_clean, layers, options);

% 使用网络进行预测
denoisedSignal = predict(net, test_noisy);

% 重塑数据回原始格式
denoisedSignal = squeeze(denoisedSignal)';
test_clean = squeeze(test_clean)';
test_noisy = squeeze(test_noisy)';

% 绘图展示结果
figure;
subplot(3,1,1);
plot(test_clean(1,:));
title('Original Clean Signal');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot(test_noisy(1,:));
title('Noisy Signal');
ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot(denoisedSignal(1,:));
title('Denoised Signal');
xlabel('Sample');
ylabel('Amplitude');
grid on;

% 计算性能指标
SNR = 10 * log10(sum(testData.^2, 2) ./ sum((testData - denoisedSignal).^2, 2));
MSE = mean((testData - denoisedSignal).^2, 'all');
NCC = sum(testData .* denoisedSignal, 'all') / sqrt(sum(testData.^2, 'all') * sum(denoisedSignal.^2, 'all'));

% 输出性能指标
fprintf('SNR: %.2f dB\n', mean(SNR));
fprintf('MSE: %.5f\n', MSE);
fprintf('NCC: %.5f\n', NCC);
