%% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

%  获取最后200行数据作为测试集（包含第4514行）
testRows = (size(data,1)-199):size(data,1);  % 即4315到4514行
testData = data(testRows, :);

% 数据集划分
trainData = data(1:1500, :);  % 使用前1500行为训练集
valData = trainData;  % 验证集可复用训练集

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 转为双精度
trainNoisy = double(trainNoisy);
trainClean = double(trainData);
testNoisy = double(testNoisy);
testClean = double(testData);

% 转置：每一列是一个样本（输入为列向量）
trainNoisy = trainNoisy';
trainClean = trainClean';
testNoisy = testNoisy';
testClean = testClean';

% 训练维度数 = 512
numDimensions = size(trainClean, 1);
SVMModels = cell(numDimensions, 1);

% 训练每一维的 SVM 模型
for dim = 1:numDimensions
    fprintf('Training SVM for dimension %d...\n', dim);
    SVMModels{dim} = fitrsvm(trainNoisy', trainClean(dim, :)', ...
        'KernelFunction', 'linear', 'Standardize', true);
end

% 对每一维进行预测，得到降噪后的信号
denoisedSignal = zeros(size(testClean));
for dim = 1:numDimensions
    fprintf('Predicting with SVM for dimension %d...\n', dim);
    denoisedSignal(dim, :) = predict(SVMModels{dim}, testNoisy');
end

% 计算平均性能指标
SNR = 10 * log10(sum(testClean.^2, 1) ./ sum((testClean - denoisedSignal).^2, 1));  % 每个样本一个SNR
avgSNR = mean(SNR);
MSE = mean((testClean - denoisedSignal).^2, 'all');
NCC = sum(testClean .* denoisedSignal, 'all') / sqrt(sum(testClean.^2, 'all') * sum(denoisedSignal.^2, 'all'));

% 输出性能指标
fprintf('Average SNR over last 200 rows: %.5f dB\n', avgSNR);
fprintf('Average MSE over last 200 rows: %.5f\n', MSE);
fprintf('Average NCC over last 200 rows: %.5f\n', NCC);
