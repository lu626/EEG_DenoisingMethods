%% 加载数据集
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');
data = EEG_all_epochs;

% 获取维度
inputSize = size(data, 2);  % 512
numChannels = 1;

% 数据集划分
numSamples = size(data, 1);
trainIdx = 1:round(numSamples * 0.8);
valIdx = (round(numSamples * 0.8) + 1):round(numSamples * 0.95);

trainData = data(trainIdx, :);
valData = data(valIdx, :);
testData = data(end-199:end, :);  %  改为最后200行作为测试集

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 重构格式为 [512,1,1,N]
train_noisy = permute(reshape(trainNoisy, [], 1, inputSize), [3 2 4 1]);
train_clean = permute(reshape(trainData,  [], 1, inputSize), [3 2 4 1]);
val_noisy   = permute(reshape(valNoisy,   [], 1, inputSize), [3 2 4 1]);
val_clean   = permute(reshape(valData,    [], 1, inputSize), [3 2 4 1]);
test_noisy  = permute(reshape(testNoisy,  [], 1, inputSize), [3 2 4 1]);
test_clean  = permute(reshape(testData,   [], 1, inputSize), [3 2 4 1]);

%% 定义 U-Net 网络结构
layers = [
    imageInputLayer([inputSize 1 1], 'Name', 'input')

    % Encoder
    convolution2dLayer([3 1], 64, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer([2 1], 'Stride', [2 1])

    convolution2dLayer([3 1], 128, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer([2 1], 'Stride', [2 1])

    % Bottleneck
    convolution2dLayer([3 1], 256, 'Padding', 'same')
    reluLayer

    % Decoder
    transposedConv2dLayer([2 1], 128, 'Stride', [2 1], 'Cropping', 'same')
    reluLayer

    transposedConv2dLayer([2 1], 64, 'Stride', [2 1], 'Cropping', 'same')
    reluLayer

    % Output
    convolution2dLayer([3 1], 1, 'Padding', 'same')
    regressionLayer
];

%% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
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

%% 使用网络预测
denoisedSignal = predict(net, test_noisy);  % [512x1x1x200]
denoisedSignal = permute(denoisedSignal, [4 3 2 1]);  % [200 x 512]
originalSignal = permute(test_clean,     [4 3 2 1]);
noisySignal    = permute(test_noisy,     [4 3 2 1]);

%  保留绘制最后一条（即第200条）测试样本
figure;
subplot(3,1,1); plot(squeeze(originalSignal(end,:))); title('Original Signal');
subplot(3,1,2); plot(squeeze(noisySignal(end,:)));    title('Noisy Signal');
subplot(3,1,3); plot(squeeze(denoisedSignal(end,:))); title('Denoised Signal');

%  修改循环：计算性能指标（对200条）
SNRs = zeros(200,1);
MSEs = zeros(200,1);
NCCs = zeros(200,1);
for i = 1:200
    clean = squeeze(originalSignal(i,:));
    den   = squeeze(denoisedSignal(i,:));
    SNRs(i) = 10 * log10(sum(clean.^2) / sum((clean - den).^2));
    MSEs(i) = mean((clean - den).^2);
    NCCs(i) = sum(clean .* den) / sqrt(sum(clean.^2) * sum(den.^2));
end

% 输出平均指标
fprintf('U-Net (Avg over last 200): SNR = %.2f dB, MSE = %.5f, NCC = %.5f\n', ...
        mean(SNRs), mean(MSEs), mean(NCCs));
