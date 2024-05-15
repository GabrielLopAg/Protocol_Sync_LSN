clear variables
close all
N = 7;
clocks = zeros(1,N);
max_offset = 200e-6;
freq_nom = 7.37e6;
T = 0.106;
freq_loc = (randn(size(clocks)) * 1e-4 + 1 ) * freq_nom;

steps = 100;
h_clocks = zeros(N, steps);
h_offsets = h_clocks;
for i = 1:steps
    h_clocks(:,i) = clocks;
    clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(1,7)-0.5);
    h_offsets(:,i) = clocks - i*T;
end
plot(h_clocks')
figure;
plot(h_offsets')