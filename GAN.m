%% 加载数据
load('EEG_all_epochs.mat');
data = EEG_all_epochs;
inputSize = size(data, 2);

% 使用前500行数据训练
maxTrainSamples = 1000;
trainData = data(1:maxTrainSamples, :);
valData = data(maxTrainSamples+1:maxTrainSamples+100, :);
testData = data(4514, :);

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 简单的数据标准化，避免训练不稳定
dataMean = mean(trainData(:));
dataStd = std(trainData(:));
trainDataNorm = (trainData - dataMean) / dataStd;
trainNoisyNorm = (trainNoisy - dataMean) / dataStd;
valDataNorm = (valData - dataMean) / dataStd;
valNoisyNorm = (valNoisy - dataMean) / dataStd;
testDataNorm = (testData - dataMean) / dataStd;
testNoisyNorm = (testNoisy - dataMean) / dataStd;

%% 改进的生成器网络 - 增加中间层宽度，使用残差连接
generatorLayers = [
    featureInputLayer(inputSize, 'Name', 'gen_input', 'Normalization', 'none')
    
    fullyConnectedLayer(1024, 'Name', 'gen_fc1')
    batchNormalizationLayer('Name', 'gen_bn1')
    reluLayer('Name', 'gen_relu1')
    
    fullyConnectedLayer(1024, 'Name', 'gen_fc2')
    batchNormalizationLayer('Name', 'gen_bn2')
    reluLayer('Name', 'gen_relu2')
    
    fullyConnectedLayer(inputSize, 'Name', 'gen_fc3')
    additionLayer(2, 'Name', 'add') % 残差连接
];
lgraph = layerGraph(generatorLayers);
lgraph = connectLayers(lgraph, 'gen_input', 'add/in2'); % 添加残差连接
generator = dlnetwork(lgraph);

%% 简化的判别器网络
discriminatorLayers = [
    featureInputLayer(inputSize, 'Name', 'disc_input', 'Normalization', 'none')
    
    fullyConnectedLayer(512, 'Name', 'disc_fc1')
    leakyReluLayer(0.2, 'Name', 'disc_leakyrelu1')
    dropoutLayer(0.3, 'Name', 'disc_drop1')
    
    fullyConnectedLayer(256, 'Name', 'disc_fc2')
    leakyReluLayer(0.2, 'Name', 'disc_leakyrelu2')
    
    fullyConnectedLayer(1, 'Name', 'disc_fc3')
    sigmoidLayer('Name', 'disc_sigmoid')
];
discriminator = dlnetwork(discriminatorLayers);

%% 优化的损失函数 - 主要关注重建质量
function [loss, gradientsGenerator] = generatorLoss(generator, discriminator, noisyData, cleanData)
    denoisedData = forward(generator, noisyData);
    
    % MSE损失
    mseLoss = mean((denoisedData - cleanData).^2, 'all');
    
    % L1损失
    l1Loss = mean(abs(denoisedData - cleanData), 'all');
    
    % 对抗损失（权重降低）
    dlYPred = forward(discriminator, denoisedData);
    adversarialLoss = -mean(log(dlYPred + eps));
    
    % 组合损失 - 主要强调重建质量
    loss = mseLoss + 0.5 * l1Loss + 0.01 * adversarialLoss;
    gradientsGenerator = dlgradient(loss, generator.Learnables);
end

function [loss, gradientsDiscriminator] = discriminatorLoss(discriminator, generator, realData, noisyData)
    generatedData = forward(generator, noisyData);
    
    dlYPredReal = forward(discriminator, realData);
    dlYPredGenerated = forward(discriminator, generatedData);
    
    % 使用更稳定的损失计算
    lossReal = -mean(log(dlYPredReal + eps));
    lossGenerated = -mean(log(1 - dlYPredGenerated + eps));
    loss = 0.5 * (lossReal + lossGenerated);
    
    gradientsDiscriminator = dlgradient(loss, discriminator.Learnables);
end

%% 训练参数优化
numEpochs = 50;
miniBatchSize = 64;  % 增大批量大小以提高稳定性
initialLearnRate = 1e-4;  % 降低学习率
numIterationsPerEpoch = floor(size(trainNoisyNorm, 1) / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

%% 训练循环
iteration = 0;
avgG = []; avgGS = [];
avgD = []; avgDS = [];

for epoch = 1:numEpochs
    idx = randperm(size(trainNoisyNorm, 1));
    trainNoisyShuffled = trainNoisyNorm(idx, :);
    trainDataShuffled = trainDataNorm(idx, :);
    
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        batchIdx = (i-1)*miniBatchSize+1:min(i*miniBatchSize, size(trainNoisyNorm, 1));
        noisyBatch = dlarray(trainNoisyShuffled(batchIdx, :)', 'CB');
        cleanBatch = dlarray(trainDataShuffled(batchIdx, :)', 'CB');
        
        % 先更新生成器多次，提高重建质量
        for j = 1:2
            [lossGen, gradientsGen] = dlfeval(@generatorLoss, generator, discriminator, noisyBatch, cleanBatch);
            [generator, avgG, avgGS] = adamupdate(generator, gradientsGen, avgG, avgGS, iteration, initialLearnRate, 0.5, 0.999);
        end
        
        % 更新判别器
        [lossDisc, gradientsDisc] = dlfeval(@discriminatorLoss, discriminator, generator, cleanBatch, noisyBatch);
        [discriminator, avgD, avgDS] = adamupdate(discriminator, gradientsDisc, avgD, avgDS, iteration, initialLearnRate, 0.5, 0.999);
    end
    
    if mod(epoch, 5) == 0
        fprintf('Epoch %d/%d - Generator Loss: %.4f, Discriminator Loss: %.4f\n', ...
            epoch, numEpochs, extractdata(lossGen), extractdata(lossDisc));
    end
end

%% 测试与评估
testNoisyNormDL = dlarray(testNoisyNorm', 'CB');
denoisedSignalNorm = extractdata(forward(generator, testNoisyNormDL))';

% 反标准化
denoisedSignal = denoisedSignalNorm * dataStd + dataMean;

% 计算性能指标
SNR = 10 * log10(sum(testData.^2) / sum((testData - denoisedSignal).^2));
MSE = mean((testData - denoisedSignal).^2);
NCC = corr(testData', denoisedSignal');

fprintf('\n性能指标:\nSNR: %.2f dB\nMSE: %.6f\nNCC: %.4f\n', SNR, MSE, NCC);

% 可视化结果
figure('Position', [100 100 800 600]);
subplot(3,1,1);
plot(testData); title('Original Signal'); ylabel('Amplitude'); grid on;
subplot(3,1,2);
plot(testNoisy); title('Noisy Signal'); ylabel('Amplitude'); grid on;
subplot(3,1,3);
plot(denoisedSignal); title('Denoised Signal'); ylabel('Amplitude'); xlabel('Sample'); grid on;
sgtitle('Re-optimized GAN-based EEG Signal Denoising');