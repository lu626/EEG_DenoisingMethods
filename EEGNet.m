%% 加载EEG数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取数据维度
[numSamples, signalLength] = size(data);  % 4514 x 512

% 使用最后200行数据作为测试集（包含第4514行）
testData = data(end-199:end, :);  % [200 x 512]
trainData = data(1:end-200, :);   % [4314 x 512]
valData = trainData;              % 验证集复用训练集

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy   = valData   + noiseAmplitude * randn(size(valData));
testNoisy  = testData  + noiseAmplitude * randn(size(testData));

% 数据重塑为 [512,1,1,N]
train_noisy = reshape(trainNoisy', [signalLength, 1, 1, size(trainNoisy, 1)]);
train_clean = reshape(trainData',  [signalLength, 1, 1, size(trainData, 1)]);
val_noisy   = reshape(valNoisy',   [signalLength, 1, 1, size(valNoisy, 1)]);
val_clean   = reshape(valData',    [signalLength, 1, 1, size(valData, 1)]);
test_noisy  = reshape(testNoisy',  [signalLength, 1, 1, size(testNoisy, 1)]);
test_clean  = reshape(testData',   [signalLength, 1, 1, size(testData, 1)]);

%% 定义EEGNet网络架构
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

%% 训练选项
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

%% 训练网络
net = trainNetwork(train_noisy, train_clean, layers, options);

%% 使用训练好的网络预测
denoisedSignal = predict(net, test_noisy);         % [512×1×1×200]
denoisedSignal = squeeze(denoisedSignal)';         % [200×512]
test_clean     = squeeze(test_clean)';             % [200×512]
test_noisy     = squeeze(test_noisy)';             % [200×512]

%% 绘图展示第1条样本（默认绘制第4514行）
figure;
subplot(3,1,1); plot(test_clean(end,:)); title('Original Clean Signal'); ylabel('Amplitude'); grid on;
subplot(3,1,2); plot(test_noisy(end,:)); title('Noisy Signal'); ylabel('Amplitude'); grid on;
subplot(3,1,3); plot(denoisedSignal(end,:)); title('Denoised Signal'); xlabel('Sample'); ylabel('Amplitude'); grid on;
sgtitle('EEGNet-based EEG Denoising (Row 4514)');

%% 逐条计算 SNR、MSE、NCC（共200条）
SNRs = zeros(200,1);
MSEs = zeros(200,1);
NCCs = zeros(200,1);
for i = 1:200
    clean = test_clean(i,:);
    den   = denoisedSignal(i,:);
    SNRs(i) = 10 * log10(sum(clean.^2) / sum((clean - den).^2));
    MSEs(i) = mean((clean - den).^2);
    NCCs(i) = sum(clean .* den) / sqrt(sum(clean.^2) * sum(den.^2));
end

%% 输出平均值 ± 标准差
SNR_avg = mean(SNRs);  SNR_std = std(SNRs);
MSE_avg = mean(MSEs);  MSE_std = std(MSEs);
NCC_avg = mean(NCCs);  NCC_std = std(NCCs);

fprintf('EEGNet [avg over last 200 rows] → SNR = %.2f ± %.2f dB, MSE = %.1f ± %.1f, NCC = %.3f ± %.3f\n', ...
    SNR_avg, SNR_std, MSE_avg, MSE_std, NCC_avg, NCC_std);
