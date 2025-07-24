%% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;
[numSamples, signalLength] = size(data);  % 4514 x 512

% ========== 数据划分 ==========
trainIdx = 1:round(numSamples * 0.8);
valIdx = (round(numSamples * 0.8) + 1):round(numSamples * 0.95);
testIdx = (numSamples - 199):numSamples;  %  最后200行

trainData = data(trainIdx, :);
valData = data(valIdx, :);
testData = data(testIdx, :);  % 200行 × 512列

% ========== 添加噪声 ==========
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy   = valData   + noiseAmplitude * randn(size(valData));
testNoisy  = testData  + noiseAmplitude * randn(size(testData));

% ========== 数据维度 reshape ==========
train_noisy = reshape(trainNoisy', [signalLength, 1, 1, size(trainNoisy, 1)]);
train_clean = reshape(trainData',   [signalLength, 1, 1, size(trainData, 1)]);
val_noisy   = reshape(valNoisy',    [signalLength, 1, 1, size(valNoisy, 1)]);
val_clean   = reshape(valData',     [signalLength, 1, 1, size(valData, 1)]);
test_noisy  = reshape(testNoisy',   [signalLength, 1, 1, size(testNoisy, 1)]);
test_clean  = reshape(testData',    [signalLength, 1, 1, size(testData, 1)]);

% ========== EEGDiR-inspired 网络 ==========
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

% ========== 训练设置 ==========
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {val_noisy, val_clean}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 5, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

% ========== 网络训练 ==========
net = trainNetwork(train_noisy, train_clean, lgraph, options);

% ========== 测试与评价 ==========
denoised = predict(net, test_noisy);     % [512×1×1×200]
denoised = squeeze(denoised)';           % [200×512]
test_clean = squeeze(test_clean)';       % [200×512]
test_noisy = squeeze(test_noisy)';       % [200×512]

%  绘图展示最后一条样本（即第4514行）
figure;
subplot(3,1,1); plot(test_clean(end,:)); title('Original Clean Signal (Row 4514)'); grid on;
subplot(3,1,2); plot(test_noisy(end,:)); title('Noisy Signal'); grid on;
subplot(3,1,3); plot(denoised(end,:));   title('Denoised Signal'); grid on;
sgtitle('EEGDiR-based EEG Denoising Result (Row 4514)');

% ========== 逐行计算评价指标 ==========
SNRs = zeros(200,1);
MSEs = zeros(200,1);
NCCs = zeros(200,1);
for i = 1:200
    clean = test_clean(i,:);
    den = denoised(i,:);
    SNRs(i) = 10 * log10(sum(clean.^2) / sum((clean - den).^2));
    MSEs(i) = mean((clean - den).^2);
    NCCs(i) = sum(clean .* den) / sqrt(sum(clean.^2) * sum(den.^2));
end

% ========== 输出平均结果 ==========
fprintf('EEGDiR (Avg over last 200): SNR = %.2f dB, MSE = %.5f, NCC = %.5f\n', ...
        mean(SNRs), mean(MSEs), mean(NCCs));
