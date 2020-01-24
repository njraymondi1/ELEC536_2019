
clc
rolloff = 0.5;  % Filter rolloff
span = 4;       % Filter span
sps = 8;        % Samples per symbol
bt = 0.5;       % time-BW product (for gaussian filter design)
M = 4;          % constellation size
k = log2(M);    % bits per symbol

% Generate the rrcos coefficients
rrcFilter = rcosdesign(rolloff, span, sps);
gFilter = gaussdesign(bt,span,sps);

figure
stem(rrcFilter)
title('RRCos Filter Coefficients, Length=16, Sps = 8, alpha=0.25')

figure
stem(gFilter)
title('Guassian FIR Filter Coefficients, TB = 0.5, Length = 16, Sps = 8')

% Generate 10000 data symbols 
data = randi([0 M-1], 10000, 1);

% Apply qam
modData = qammod(data, M);
tic
% upsample and filter the input data.
txSig = upfirdn(modData, rrcFilter, sps);
txSig2 = upfirdn(modData, gFilter, sps);

% Filter and downsample the received signal, shift to account for filter
% delay
txFilt = upfirdn(txSig, rrcFilter, 1, sps);
txFilt = txFilt(span+1:end-span);
txFilt2 = upfirdn(txSig2, gFilter, 1, sps);
txFilt2 = txFilt2(span+1:end-span);
toc
figure
subplot(2,1,1)
stem(modData(1:50))
title('Modulated Data')
subplot(2,1,2)
stem(upsample(modData(1:50),sps))
title('RRCOS Filtered Upsampled Mod Data, 8 samples/symbol, Length 16')
hold on
stem(txSig(sps*span/2 + 1:50*sps+sps*span/2))%sps*span:50*sps+sps*span))
%plot(1:length(txSig(2*span+1:50*sps+2*span)),txSig(2*span+1:50*sps+2*span))
%legend('Upsampled Modulated Signal','Filtered Upsampled Modulated Signal')

figure
subplot(2,1,1)
stem(modData(1:50))
title('Modulated Data')
subplot(2,1,2)
stem(upsample(modData(1:50),sps))
title('Gaussian Filtered Upsampled Mod Data, 8 samples/symbol, Length 16')
hold on
stem(txSig2(sps*span/2 + 1:50*sps+sps*span/2))

fftData = fft(txSig);
figure
plot(abs(fftData))

figure
stem(modData(1:50))
hold on
plot(1:50,txFilt(1:50))
legend('Modulated Data','Downsampled Filtered Signal')
title('Upsampled, Filtered, Filtered, DownSampled')

figure
pwelch(txSig,[],[],[],'centered')
title('Welch PSD of Modulated, Upsampled, Filtered Signal')













