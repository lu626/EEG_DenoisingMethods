tic;  % >>> 开始记录 Processing Time <<<

%% ——— 前处理保持不变 ———
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat'); 
data = EEG_all_epochs;
trainIdx = 1:round(size(data,1)*0.8); 
valIdx = round(size(data,1)*0.8)+1:round(size(data,1)*0.95); 
trainData = data(trainIdx,:); 
valData = data(valIdx,:); 
noiseAmplitude = 50;

trainNoisy = trainData + noiseAmplitude * randn(size(trainData));
valNoisy   = valData   + noiseAmplitude * randn(size(valData));

train_noisy = reshape(trainNoisy', [size(trainData,2),1,1,size(trainNoisy,1)]);
train_clean = reshape(trainData',   [size(trainData,2),1,1,size(trainNoisy,1)]);
val_noisy   = reshape(valNoisy',   [size(valNoisy,2),1,1,size(valNoisy,1)]);
val_clean   = reshape(valData',     [size(valData,2),1,1,size(valNoisy,1)]);

%% ——— TADA autoencoder 架构 ———
layers = [
  imageInputLayer([size(trainData,2) 1 1],'Name','input')

  convolution2dLayer([5 1],64,'Padding','same','Name','enc_conv1')
  reluLayer('Name','enc_relu1')
  maxPooling2dLayer([2 1],'Stride',[2 1],'Name','enc_pool1')

  convolution2dLayer([5 1],128,'Padding','same','Name','enc_conv2')
  reluLayer('Name','enc_relu2')
  maxPooling2dLayer([2 1],'Stride',[2 1],'Name','enc_pool2')

  transposedConv2dLayer([4 1],64,'Stride',[2 1],'Cropping','same','Name','dec_conv1')
  reluLayer('Name','dec_relu1')
  transposedConv2dLayer([4 1],1,'Stride',[2 1],'Cropping','same','Name','dec_conv2')

  regressionLayer('Name','output')
];

lgraph = layerGraph(layers);

options = trainingOptions('adam', ...
    'MaxEpochs',40, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',1e-3, ...
    'ValidationData',{val_noisy,val_clean}, ...
    'Plots','training-progress');

net = trainNetwork(train_noisy, train_clean, lgraph, options);

param_count = 0;
for i = 1:numel(net.Layers)
    if isprop(net.Layers(i), 'Weights') && ~isempty(net.Layers(i).Weights)
        param_count = param_count + numel(net.Layers(i).Weights);
    end
    if isprop(net.Layers(i), 'Bias') && ~isempty(net.Layers(i).Bias)
        param_count = param_count + numel(net.Layers(i).Bias);
    end
end
param_count = param_count / 1e6;  % 单位：M


%% ——— Inference Time：对200个样本前向传播测平均时间（ms）———
numInfer = 200;
inferTimes = zeros(1, numInfer);
for i = 1:numInfer
    sample = reshape(data(end-i+1,:)', [512 1 1 1]);
    tStart = tic;
    predict(net, sample);
    inferTimes(i) = toc(tStart);
end
avg_infer_time_ms = mean(inferTimes) * 1000;

%% ——— 测试部分（✅ 最后200行）———
numTest = 200;
testData = double(data(end-numTest+1:end,:));  % ← 修复：确保为 double 类型
testNoisy = testData + noiseAmplitude * randn(size(testData));  % ← 修复：确保维度一致

SNR_list = zeros(1, numTest);
MSE_list = zeros(1, numTest);
NCC_list = zeros(1, numTest);

for i = 1:numTest
    noisy_sample = reshape(testNoisy(i,:)', [size(testData,2),1,1,1]);
    clean_sample = testData(i,:)';
    denoised = predict(net, noisy_sample);
    den = squeeze(denoised)';

    SNR_list(i) = 10*log10(sum(clean_sample.^2)/sum((clean_sample-den').^2));
    MSE_list(i) = mean((clean_sample-den').^2);
    NCC_list(i) = sum(clean_sample.*den')/sqrt(sum(clean_sample.^2)*sum(den'.^2));
end

%% ——— 输出最终指标（带标准差）———
SNR_avg = mean(SNR_list);
SNR_std = std(SNR_list);

MSE_avg = mean(MSE_list);
MSE_std = std(MSE_list);

NCC_avg = mean(NCC_list);
NCC_std = std(NCC_list);

fprintf('TADA [avg over last 200 rows] → SNR = %.2f ± %.2f dB, MSE = %.1f ± %.1f, NCC = %.3f ± %.3f\n', ...
    SNR_avg, SNR_std, MSE_avg, MSE_std, NCC_avg, NCC_std);

%% ——— 图像显示：✅ 使用最后一行数据（第4514行）绘图 ———
figure;
subplot(3,1,1); plot(testData(end,:)); title('Original'); grid on;
subplot(3,1,2); plot(testNoisy(end,:)); title('Noisy'); grid on;
subplot(3,1,3); 
denoised4514 = predict(net, reshape(testNoisy(end,:)',[512 1 1 1]));
plot(squeeze(denoised4514)); 
title('TADA Denoised'); grid on;
sgtitle('TADA-based EEG Denoising (Row 4514)');

%% ——— 最终统计输出（参数量 + 推理时间 + 总耗时）———
total_time = toc;
fprintf('Parameter (M): %.2f\n', param_count);
fprintf('Inference Time (ms): %.2f\n', avg_infer_time_ms);
fprintf('Processing Time (s): %.2f\n', total_time);
