
% Nate Raymondi
% Adapted with help from MathWorks "passband demo" 
% https://www.mathworks.com/help/comm/examples/...
%   passband-modulation-with-adjacent-channel-interference.html

Fc = 2.5e6;         
Rsym = 1e6;        
sps = 8;                % Number of samples per symbol
span = 4;
frameLength = 2048; 
M = 4;                  % Modulation order (4-QAM)
Fs = Rsym * sps;

% Create a 16-QAM modulator.
qamMod = comm.RectangularQAMModulator(M);
b = randi([0 M-1], frameLength, 1);
txSym = qamMod(b);

rctFilt = comm.RaisedCosineTransmitFilter('RolloffFactor', 0.5, ...
  'OutputSamplesPerSymbol', sps, ...
  'FilterSpanInSymbols', span);
x = rctFilt(txSym);

figure
stem(x(sps*span/2 + 1:20*sps+sps*span/2))
hold on
stem(upsample(txSym(1:20),sps))
title('RRCOS Filtered Upsampled Mod Data, Alpha = 0.5')

figure
subplot(2,1,1)
pwelch(x,hamming(512),[],[],Fs,'centered')
title({'Welch PSD of Modulated, Upsampled, Filtered Signal',...
    'Filter Order = 4*8, Alpha = 0.75'})

t = (0:1/Fs:(frameLength/Rsym)-1/Fs).';
carrier = sqrt(2)*exp(1i*2*pi*Fc*t);
xUp = real(x.*carrier);
%figure
subplot(2,1,2)
pwelch(xUp,hamming(512),[],[],Fs,'centered')
title({'Welch PSD of Upconverted to 2.5MHz, Modulated, Upsampled, Filtered Signal',...
    'Filter Order = 4*8, Alpha = 0.5'})

xDown = xUp.*conj(carrier);

figure
subplot(2,1,1)
pwelch(x,hamming(512),[],[],Fs,'centered')
title({'Welch PSD of Modulated, Upsampled, Filtered Signal',...
    'Filter Order = 4*8, Alpha = 0.5'})
subplot(2,1,2)
pwelch(xDown,hamming(512),[],[],Fs,'centered')
title({'Welch PSD of Upconverted to 2.5MHz then Downconverted Signal',...
    'Filter Order = 4*8, Alpha = 0.5'})

rcrFiltRX = comm.RaisedCosineReceiveFilter('RolloffFactor',0.5, ...
  'FilterSpanInSymbols',span,'InputSamplesPerSymbol',sps, ...
  'DecimationFactor',8);

xFiltRx = rcrFiltRX(xDown);

figure
stem(txSym(1:30))
hold on
plot(1:30,xFiltRx(1+span:30+span))
title('Downconverted, Downsample, Recevied Filtered Signal')
legend('Oringinal Data Symbols', 'Received Signal')











