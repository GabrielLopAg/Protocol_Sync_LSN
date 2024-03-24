num_nodes = 10; % Número de nodos
sim_duration = 3600; 
time_step = 0.01; % Duración en el que el reloj es estable
mean_sender_delay = 100e-6; % Mean Tx delay (seg)
std_sender_delay = 11e-6; % Standard deviation Tx delay (seg)
mean_receiver_delay = 100e-6; % Mean receiver delay (seg)
std_receiver_delay = 11e-6; % Standard deviation of receiver delay (seg)
crystal_freq_mean = 7.3738e6; % Mean crystal oscillator frequency (Hz)
crystal_freq_std = 100e3; % Standard deviation of crystal oscillator frequency (Hz)

time = 0:time_step:sim_duration;
clock_drift = zeros(num_nodes, length(time));
sender_delay = normrnd(mean_sender_delay, std_sender_delay, num_nodes, length(time));
receiver_delay = normrnd(mean_receiver_delay, std_receiver_delay, num_nodes, length(time));
crystal_freq = normrnd(crystal_freq_mean, crystal_freq_std, num_nodes, 1);

for node = 1:num_nodes
    drift_ppm = ((crystal_freq(node) - crystal_freq_mean) / crystal_freq_mean) * 1e6; % ppm del drift
    drift_std = drift_ppm / 1e6 * crystal_freq_mean; % Desviación estandar del drift
    drift = normrnd(0, drift_std, 1, length(time)); 
    clock_drift(node, :) = drift;
end

time_deviation = zeros(num_nodes, length(time));
for node = 1:num_nodes
    for t = 2:length(time)
        time_deviation(node, t) = time_deviation(node, t-1) + clock_drift(node, t) * time_step;
    end
end
