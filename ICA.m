%% ===============================
%  ICA: Heatmaps + Best Channel + Single Evaluation on Row 4514
%  (Non-generalizable method; no averaging over last rows)
% ===============================
clear; clc; close all;
tic;  % processing time

%% ---- Load data
load('Q:\APP\EEGdenoiseNet-master\EEGdenoiseNet-master\data\EEG_all_epochs.mat');

%% ---- Grid settings
noise_amplitudes = 5:5:50;                          % A = 5:5:50
channel_numbers  = [1,50,100,150,200,250,300,350,400,450,500];

SNR_results = zeros(length(noise_amplitudes), length(channel_numbers));
MSE_results = zeros(length(noise_amplitudes), length(channel_numbers));
NCC_results = zeros(length(noise_amplitudes), length(channel_numbers));

target_idx = 4514;   % evaluation target row
rng(0);              % reproducibility

%% ---- Main loops (heatmaps)
for c = 1:length(channel_numbers)
    num_channels = channel_numbers(c);

    % Assemble multichannel matrix including target_idx
    if num_channels == 1
        base_data = EEG_all_epochs(target_idx, :);
    else
        pool = setdiff(1:size(EEG_all_epochs,1), target_idx);
        rand_idx = pool(randperm(numel(pool), num_channels-1));
        base_data = [EEG_all_epochs(target_idx, :); EEG_all_epochs(rand_idx, :)];
    end

    for i = 1:length(noise_amplitudes)
        A = noise_amplitudes(i);
        noisy_data = base_data + A * randn(size(base_data));

        % ===== ICA denoising =====
        if num_channels > 1
            % FastICA
            [S, A_mix, ~] = fastica(noisy_data', ...
                                    'approach','symm', ...
                                    'g','tanh', ...
                                    'maxNumIterations',500, ...
                                    'epsilon',1e-3);

            % Heuristic component selection by kurtosis (keep non-Gaussian)
            kc = kurtosis(S', 0);                    % 1×ncomp
            keep = kc > 3.2;                         % threshold
            if ~any(keep), keep(:) = true; end       % fallback: keep all

            % Reconstruct from selected components
            clean_data = (A_mix(:,keep) * S(keep,:))';
        else
            clean_data = noisy_data;                 % single channel: pass-through
        end

        % ----- Metrics (per Eq. 6/7) on the target channel (row 1)
        X = base_data(1,:);              % clean reference segment (target_idx)
        Z = clean_data(1,:);             % denoised
        % SNR: 10*log10( sum(X.^2) / sum((X-Z).^2) )
        SNR_results(i,c) = 10*log10( sum(X.^2) / sum((X - Z).^2) );
        % MSE
        MSE_results(i,c) = mean( (X - Z).^2 );
        % NCC (zero-mean normalized correlation)
        x0 = X - mean(X); z0 = Z - mean(Z);
        NCC_results(i,c) = sum(x0 .* z0) / sqrt( sum(x0.^2) * sum(z0.^2) );
    end
end

%% ---- Plot heatmaps
figure;

subplot(3,1,1);
h1 = heatmap(channel_numbers, noise_amplitudes, SNR_results);
h1.Title  = 'SNR Heatmap';
h1.XLabel = 'Channel Counts (ICA)';   % space before '('
h1.YLabel = 'Noise Amplitude';

subplot(3,1,2);
h2 = heatmap(channel_numbers, noise_amplitudes, MSE_results);
h2.Title  = 'MSE Heatmap';
h2.XLabel = 'Channel Counts (ICA)';
h2.YLabel = 'Noise Amplitude';

subplot(3,1,3);
h3 = heatmap(channel_numbers, noise_amplitudes, NCC_results);
h3.Title  = 'NCC Heatmap';
h3.XLabel = 'Channel Counts (ICA)';
h3.YLabel = 'Noise Amplitude';

%% ---- Choose best channel count at A = 50 (last row of heatmap)
[~, best_c_idx]   = max(SNR_results(end,:));
best_channel_num  = channel_numbers(best_c_idx);

%% ---- Single evaluation on Row 4514 @ A=50 @ best_channel_num
if best_channel_num == 1
    data_multi = EEG_all_epochs(target_idx, :);
else
    pool = setdiff(1:size(EEG_all_epochs,1), target_idx);
    rand_idx = pool(randperm(numel(pool), best_channel_num-1));
    data_multi = [EEG_all_epochs(target_idx, :); EEG_all_epochs(rand_idx, :)];
end

A = 50;
noisy_multi = data_multi + A * randn(size(data_multi));

if best_channel_num > 1
    [S, A_mix, ~] = fastica(noisy_multi', ...
                            'approach','symm', ...
                            'g','tanh', ...
                            'maxNumIterations',500, ...
                            'epsilon',1e-3);
    kc   = kurtosis(S', 0);
    keep = kc > 3.2;
    if ~any(keep), keep(:) = true; end
    demo_clean = (A_mix(:,keep) * S(keep,:))';
else
    demo_clean = noisy_multi;
end

X = data_multi(1,:);
Z = demo_clean(1,:);
Y = noisy_multi(1,:);   % for plotting only

final_SNR = 10*log10( sum(X.^2) / sum((X - Z).^2) );
final_MSE = mean( (X - Z).^2 );
x0 = X - mean(X); z0 = Z - mean(Z);
final_NCC = sum(x0 .* z0) / sqrt( sum(x0.^2) * sum(z0.^2) );

%% ---- Plots (Row 4514)
figure;
subplot(3,1,1); plot(X); title('Original Signal (Row 4514)'); ylabel('Amplitude');
subplot(3,1,2); plot(Y); title(sprintf('Noisy Signal (A = %d, Channels = %d)', A, best_channel_num)); ylabel('Amplitude');
subplot(3,1,3); plot(Z); title('ICA Denoised Signal'); xlabel('Time Points'); ylabel('Amplitude');

%% ---- Print final single-sample results
fprintf('\n=== Final Result (ICA, Row 4514 @ A=50, Best Channels = %d) ===\n', best_channel_num);
fprintf('SNR: %.4f dB\n', final_SNR);
fprintf('MSE: %.4f\n',  final_MSE);
fprintf('NCC: %.4f\n',  final_NCC);

%% ---- Stats
param_count = 0;           % ICA (here) has no trainable params for evaluation
processing_time = toc;
fprintf('ICA Parameter Count: %d\n', param_count);
fprintf('ICA Processing Time: %.4f seconds\n', processing_time);
