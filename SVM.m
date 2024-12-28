% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取第4514行的数据
eeg_data = data(4514, :);  % 选择第4514行数据

% 数据集划分
numSamples = size(data, 1);  % 数据集的总行数（4514行）
trainData = data(1:1500, :);  % 使用前1500行作为训练集
valData = eeg_data;  % 使用第4514行作为验证集
testData = eeg_data;  % 使用第4514行作为测试集

% 添加噪声
noiseAmplitude = 50;  % 噪声幅度为50
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 确保数据是双精度浮点数
trainNoisy = double(trainNoisy);
trainClean = double(trainData);

valNoisy = double(valNoisy);
valClean = double(valData);

testNoisy = double(testNoisy);
testClean = double(testData);

% 转置数据，使每一列表示一个样本，每一行表示一个特征
trainNoisy = trainNoisy';  % 每一列是一个样本
trainClean = trainClean';

valNoisy = valNoisy';
valClean = valClean';

testNoisy = testNoisy';
testClean = testClean';

% 初始化变量存储每一维的模型
numDimensions = size(trainClean, 1); % 每一行表示一个维度
SVMModels = cell(numDimensions, 1); % 存储多个 SVM 模型

% 逐维训练 SVM
for dim = 1:numDimensions
    fprintf('Training SVM for dimension %d...\n', dim);
    SVMModels{dim} = fitrsvm(trainNoisy', trainClean(dim, :)', ...
        'KernelFunction', 'linear', 'Standardize', true);
end

% 使用 SVM 逐维进行预测（降噪）
denoisedSignal = zeros(size(testClean));
for dim = 1:numDimensions
    fprintf('Predicting with SVM for dimension %d...\n', dim);
    denoisedSignal(dim, :) = predict(SVMModels{dim}, testNoisy');
end

% 绘图
figure;

% 原始信号图（使用第4514行的数据）
subplot(3, 1, 1);
plot(testClean);
title('Original Signal');
ylabel('Amplitude');

% 加噪声后的信号图（使用第4514行的数据）
subplot(3, 1, 2);
plot(testNoisy);
title('Noisy Signal');
ylabel('Amplitude');

% 降噪后的信号图（使用第4514行的数据）
subplot(3, 1, 3);
plot(denoisedSignal);
title('Denoised Signal (SVM)');
ylabel('Amplitude');

% 计算性能指标
SNR = 10 * log10(sum(testClean.^2, 2) ./ sum((testClean - denoisedSignal).^2, 2));
MSE = mean((testClean - denoisedSignal).^2, 'all');
NCC = sum(testClean .* denoisedSignal, 'all') / sqrt(sum(testClean.^2, 'all') * sum(denoisedSignal.^2, 'all'));

% 输出性能指标
fprintf('SNR: %.5f dB\n', mean(SNR));
fprintf('MSE: %.5f\n', MSE);
fprintf('NCC: %.5f\n', NCC);
