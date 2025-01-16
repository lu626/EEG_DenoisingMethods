% 加载EEG数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取数据维度
[numSamples, signalLength] = size(data);  % 4514 x 512

% 选择第4514行进行测试
testData = data(4514, :);

% 创建训练和验证集（用于训练的信号不包含第4514行）
trainData = data(1:4513, :);
valData = trainData;  % 可以使用训练集的部分作为验证集

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

% 定义EEGNet网络架构
layers = [
    imageInputLayer([signalLength 1 1], 'Name', 'input')
    
    % 第一层卷积
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    % 第二层卷积
    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    % 第三层卷积
    convolution2dLayer([3 1], 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % 第四层卷积
    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    % 第五层卷积
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    
    % 最后一层卷积，输出一个通道
    convolution2dLayer([3 1], 1, 'Padding', 'same', 'Name', 'conv6')
    regressionLayer('Name', 'output')
];

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
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

% 使用训练好的网络进行预测（仅对第4514行测试数据进行预测）
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
SNR = 10 * log10(sum(test_clean.^2, 2) ./ sum((test_clean - denoisedSignal).^2, 2));
MSE = mean((test_clean - denoisedSignal).^2, 'all');
NCC = sum(test_clean .* denoisedSignal, 'all') / sqrt(sum(test_clean.^2, 'all') * sum(denoisedSignal.^2, 'all'));

% 输出性能指标
fprintf('Average Signal-to-Noise Ratio (SNR): %.2f dB\n', mean(SNR));
fprintf('Mean Squared Error (MSE): %.5f\n', MSE);
fprintf('Normalized Cross-Correlation (NCC): %.5f\n', NCC);
