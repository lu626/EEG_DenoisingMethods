%% 加载数据
load('EEG_all_epochs.mat');
data = EEG_all_epochs;
inputSize = size(data, 2);

% 使用前1000行数据训练
maxTrainSamples = 1000;
trainData = data(1:maxTrainSamples, :);
valData = data(maxTrainSamples+1:maxTrainSamples+100, :);

% 使用最后200行为测试集（包含第4514行）
testRows = (size(data, 1) - 199):size(data, 1);
testData = data(testRows, :);

% 添加噪声
noiseAmplitude = 50;
trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy = valData + noiseAmplitude * randn(size(valData));
testNoisy = testData + noiseAmplitude * randn(size(testData));

% 数据标准化
dataMean = mean(trainData(:));
dataStd = std(trainData(:));
trainDataNorm = (trainData - dataMean) / dataStd;
trainNoisyNorm = (trainNoisy - dataMean) / dataStd;
valDataNorm = (valData - dataMean) / dataStd;
valNoisyNorm = (valNoisy - dataMean) / dataStd;
testDataNorm = (testData - dataMean) / dataStd;
testNoisyNorm = (testNoisy - dataMean) / dataStd;

%% 构建生成器网络（含残差连接）
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
lgraph = connectLayers(lgraph, 'gen_input', 'add/in2');
generator = dlnetwork(lgraph);

%% 判别器网络
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

%% 损失函数定义
function [loss, gradientsGenerator] = generatorLoss(generator, discriminator, noisyData, cleanData)
    denoisedData = forward(generator, noisyData);
    mseLoss = mean((denoisedData - cleanData).^2, 'all');
    l1Loss = mean(abs(denoisedData - cleanData), 'all');
    dlYPred = forward(discriminator, denoisedData);
    adversarialLoss = -mean(log(dlYPred + eps));
    loss = mseLoss + 0.5 * l1Loss + 0.01 * adversarialLoss;
    gradientsGenerator = dlgradient(loss, generator.Learnables);
end

function [loss, gradientsDiscriminator] = discriminatorLoss(discriminator, generator, realData, noisyData)
    generatedData = forward(generator, noisyData);
    dlYPredReal = forward(discriminator, realData);
    dlYPredGenerated = forward(discriminator, generatedData);
    lossReal = -mean(log(dlYPredReal + eps));
    lossGenerated = -mean(log(1 - dlYPredGenerated + eps));
    loss = 0.5 * (lossReal + lossGenerated);
    gradientsDiscriminator = dlgradient(loss, discriminator.Learnables);
end

%% 训练参数
numEpochs = 50;
miniBatchSize = 64;
initialLearnRate = 1e-4;
numIterationsPerEpoch = floor(size(trainNoisyNorm, 1) / miniBatchSize);

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

        % 更新生成器
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

%% 测试与评估：使用最后200行测试集
denoisedAll = zeros(size(testData));
for i = 1:size(testData, 1)
    testNoisyNormDL = dlarray(testNoisyNorm(i, :)', 'CB');
    denoisedNorm = extractdata(forward(generator, testNoisyNormDL))';
    denoisedAll(i, :) = denoisedNorm * dataStd + dataMean;
end

%% 评估性能指标（200行，带标准差）
SNR = 10 * log10(sum(testData.^2, 2) ./ sum((testData - denoisedAll).^2, 2));
MSE = mean((testData - denoisedAll).^2, 2);
NCC = sum(testData .* denoisedAll, 2) ./ sqrt(sum(testData.^2, 2) .* sum(denoisedAll.^2, 2));

SNR_avg = mean(SNR);  SNR_std = std(SNR);
MSE_avg = mean(MSE);  MSE_std = std(MSE);
NCC_avg = mean(NCC);  NCC_std = std(NCC);

fprintf('\nGAN [avg over last 200 rows] → SNR = %.2f ± %.2f dB, MSE = %.1f ± %.1f, NCC = %.3f ± %.3f\n', ...
    SNR_avg, SNR_std, MSE_avg, MSE_std, NCC_avg, NCC_std);

%% 绘图展示（第4514行）
figure('Position', [100 100 800 600]);
idx = size(testData, 1); % 即第4514行
subplot(3,1,1);
plot(testData(idx, :)); title('Original Signal (Row 4514)'); ylabel('Amplitude'); grid on;
subplot(3,1,2);
plot(testNoisy(idx, :)); title('Noisy Signal'); ylabel('Amplitude'); grid on;
subplot(3,1,3);
plot(denoisedAll(idx, :)); title('Denoised Signal'); ylabel('Amplitude'); xlabel('Sample'); grid on;
sgtitle('GAN-based EEG Denoising Result (Row 4514)');
