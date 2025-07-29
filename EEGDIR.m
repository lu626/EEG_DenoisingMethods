clear; clc; close all;
tic; % Start total processing time

%% 加载数据
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;
[numSamples, signalLength] = size(data);

% 数据划分
trainIdx = 1:round(numSamples * 0.8);
valIdx = (round(numSamples * 0.8) + 1):round(numSamples * 0.95);
testIdx = (numSamples - 199):numSamples;

trainData = data(trainIdx, :);
valData   = data(valIdx, :);
testData  = data(testIdx, :);

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy   = valData   + noiseAmplitude * randn(size(valData));
testNoisy  = testData  + noiseAmplitude * randn(size(testData));

% 数据 reshape 为 [512,1,1,N]
train_noisy = reshape(trainNoisy', [signalLength, 1, 1, size(trainNoisy, 1)]);
train_clean = reshape(trainData',   [signalLength, 1, 1, size(trainData, 1)]);
val_noisy   = reshape(valNoisy',    [signalLength, 1, 1, size(valNoisy, 1)]);
val_clean   = reshape(valData',     [signalLength, 1, 1, size(valData, 1)]);
test_noisy  = reshape(testNoisy',   [signalLength, 1, 1, size(testNoisy, 1)]);
test_clean  = reshape(testData',    [signalLength, 1, 1, size(testData, 1)]);

%% 定义 EEGDiR 网络结构（含残差连接）
layers = [
    imageInputLayer([signalLength 1 1], 'Name', 'input')
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    batchNormalizationLayer('Name', 'bn1')
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    batchNormalizationLayer('Name', 'bn2')
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    batchNormalizationLayer('Name', 'bn3')
    additionLayer(2, 'Name', 'add')
    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv4')
    reluLayer('Name', 'relu4')
    batchNormalizationLayer('Name', 'bn4')
    convolution2dLayer([3 1], 1, 'Padding', 'same', 'Name', 'conv_out')
    regressionLayer('Name', 'output')
];
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'bn1', 'add/in2');

%% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {val_noisy, val_clean}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% 网络训练
net = trainNetwork(train_noisy, train_clean, lgraph, options);

totalParams = 0;
layers = net.Layers;
for i = 1:numel(layers)
    if isprop(layers(i), 'Weights') && ~isempty(layers(i).Weights)
        totalParams = totalParams + numel(layers(i).Weights);
    end
    if isprop(layers(i), 'Bias') && ~isempty(layers(i).Bias)
        totalParams = totalParams + numel(layers(i).Bias);
    end
end
fprintf('\n模型参数量（可训练参数）: %.4f M（%.0f）\n', totalParams / 1e6, totalParams);


%% Inference Time（200 次平均）
inferenceTimes = zeros(200,1);
for i = 1:200
    sample = test_noisy(:,:,:,i);  % 注意不要 squeeze，保持 [512 1 1]
    tStart = tic;
    predict(net, sample);
    inferenceTimes(i) = toc(tStart);
end
avgInferenceTime = mean(inferenceTimes);
fprintf('平均 Inference Time（秒）: %.6f（%.2f 毫秒）\n', avgInferenceTime, avgInferenceTime * 1000);

%% 模型预测（整批测试）
denoised = predict(net, test_noisy);     % [512×1×1×200]
denoised = squeeze(denoised)';           % [200×512]
test_clean = squeeze(test_clean)';       % [200×512]
test_noisy = squeeze(test_noisy)';       % [200×512]

%% 绘图展示第4514行
figure;
subplot(3,1,1); plot(test_clean(end,:)); title('Original Clean Signal (Row 4514)'); grid on;
subplot(3,1,2); plot(test_noisy(end,:)); title('Noisy Signal'); grid on;
subplot(3,1,3); plot(denoised(end,:));   title('Denoised Signal'); grid on;
sgtitle('EEGDiR-based EEG Denoising (Row 4514)');

%% 逐行计算 SNR、MSE、NCC
SNRs = zeros(200,1);
MSEs = zeros(200,1);
NCCs = zeros(200,1);
for i = 1:200
    clean = test_clean(i,:);
    den = denoised(i,:);
    SNRs(i) = 10 * log10(sum(clean.^2) / sum((clean - den).^2));
    MSEs(i) = mean((clean - den).^2);
    NCCs(i) = sum(clean .* den) / (sqrt(sum(clean.^2)) * sqrt(sum(den.^2)));
end

% 输出结果
fprintf('\nEEGDiR [avg over last 200 rows]:\n');
fprintf('SNR = %.2f ± %.2f dB\n', mean(SNRs), std(SNRs));
fprintf('MSE = %.1f ± %.1f\n', mean(MSEs), std(MSEs));
fprintf('NCC = %.3f ± %.3f\n', mean(NCCs), std(NCCs));

% 输出总耗时
totalTime = toc;
fprintf('总 Processing Time（秒）: %.2f\n', totalTime);
