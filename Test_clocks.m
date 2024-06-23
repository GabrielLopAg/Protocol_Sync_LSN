clear variables
close all
N = 70;
clocks = zeros(1,N);
% max_offset = 200e-6;
freq_nom = 7.37e6;
T = 0.001;
freq_loc = (randn(size(clocks)) * 1e-6 + 1 ) * freq_nom;
std = 1e-6;
steps = 1*1000;
h_clocks = zeros(N, steps);
h_offsets = h_clocks;
for i = 1:steps
    h_clocks(:,i) = clocks;
    freq_loc = (sqrt(T)*randn(size(clocks)) * std + 1 ) .* freq_loc;
    clocks = clocks + T*freq_loc/freq_nom;
    h_offsets(:,i) = clocks - i*T;
end
plot(h_clocks')
title('Relojes')
figure;
plot(h_offsets')
title("Offsets ")
annotation('textbox',[0.15 0.6 0.3 0.3], 'String', ...
   ["\Delta t = "+max(h_offsets,[],'all'); "\sigma = "+std], ...
   'FitBoxToText', 'on');