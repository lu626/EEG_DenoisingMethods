tic;  % >>> 记录 Processing Time <<<

%% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取最后200行数据作为测试集（4315~4514）
testRows = (size(data,1)-199):size(data,1);
testData = data(testRows, :);

% 数据集划分
trainData = data(1:1500, :);  % 使用前1500行为训练集
valData = trainData;  % 验证集可复用训练集

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy   = valData   + noiseAmplitude * randn(size(valData));
testNoisy  = testData  + noiseAmplitude * randn(size(testData));

% 转为双精度
trainNoisy  = double(trainNoisy);
trainClean  = double(trainData);
testNoisy   = double(testNoisy);
testClean   = double(testData);

% 转置：每一列是一个样本（列向量为样本）
trainNoisy = trainNoisy';
trainClean = trainClean';
testNoisy  = testNoisy';
testClean  = testClean';

% 每一维建立一个 SVM 模型
numDimensions = size(trainClean, 1);  % 512
SVMModels = cell(numDimensions, 1);

for dim = 1:numDimensions
    fprintf('Training SVM for dimension %d...\n', dim);
    SVMModels{dim} = fitrsvm(trainNoisy', trainClean(dim, :)', ...
        'KernelFunction', 'linear', 'Standardize', true);
end

%% Inference Time 计算（200个样本）
numSamples = size(testNoisy, 2);  % 200
inferTimes = zeros(1, numSamples);
for i = 1:numSamples
    tStart = tic;
    for dim = 1:numDimensions
        predict(SVMModels{dim}, testNoisy(:, i)');
    end
    inferTimes(i) = toc(tStart);
end
avg_infer_time_ms = mean(inferTimes) * 1000;

% 逐维预测，重建去噪信号
denoisedSignal = zeros(size(testClean));
for dim = 1:numDimensions
    fprintf('Predicting with SVM for dimension %d...\n', dim);
    denoisedSignal(dim, :) = predict(SVMModels{dim}, testNoisy');
end

% 转置回来，统一为 [200 x 512]
testClean = testClean';
denoisedSignal = denoisedSignal';

%% 计算性能指标（SNR、MSE、NCC）每行一个样本
SNR_list = zeros(200, 1);
MSE_list = zeros(200, 1);
NCC_list = zeros(200, 1);
for i = 1:200
    clean = testClean(i, :);
    den   = denoisedSignal(i, :);
    SNR_list(i) = 10 * log10(sum(clean.^2) / sum((clean - den).^2));
    MSE_list(i) = mean((clean - den).^2);
    NCC_list(i) = sum(clean .* den) / sqrt(sum(clean.^2) * sum(den.^2));
end

% 输出平均值 ± 标准差
SNR_avg = mean(SNR_list);  SNR_std = std(SNR_list);
MSE_avg = mean(MSE_list);  MSE_std = std(MSE_list);
NCC_avg = mean(NCC_list);  NCC_std = std(NCC_list);

fprintf('SVM [avg over last 200 rows] → SNR = %.2f ± %.2f dB, MSE = %.1f ± %.1f, NCC = %.3f ± %.3f\n', ...
    SNR_avg, SNR_std, MSE_avg, MSE_std, NCC_avg, NCC_std);

%% 最终统计输出（SVM无参数量）
total_time = toc;
fprintf('Parameter (M): —— (N/A for SVM)\n');
fprintf('Inference Time (ms): %.2f\n', avg_infer_time_ms);
fprintf('Processing Time (s): %.2f\n', total_time);
