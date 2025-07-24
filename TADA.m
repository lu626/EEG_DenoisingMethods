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

%% ——— 测试部分（ 最后200行）———
numTest = 200;
testData = data(end-numTest+1:end,:);
testNoisy = testData + noiseAmplitude*randn(size(testData));

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

SNR_avg = mean(SNR_list);
MSE_avg = mean(MSE_list);
NCC_avg = mean(NCC_list);

%% ——— 图像显示： 使用最后一行数据（第4514行）绘图 ———
figure;
subplot(3,1,1); plot(testData(end,:)); title('Original'); grid on;
subplot(3,1,2); plot(testNoisy(end,:)); title('Noisy'); grid on;
subplot(3,1,3); plot(squeeze(predict(net, reshape(testNoisy(end,:)',[512 1 1 1])))'); title('TADA Denoised'); grid on;
sgtitle('TADA-based EEG Denoising (Row 4514)');

%% ——— 输出结果 ———
fprintf('TADA [avg over last 200 rows] → SNR = %.2f dB, MSE = %.5f, NCC = %.5f\n', SNR_avg, MSE_avg, NCC_avg);
