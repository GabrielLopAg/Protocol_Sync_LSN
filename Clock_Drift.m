num_nodes = 10;
sampling_rate = 1; % Hz
duration = 3600; % segundos
drift_rate = 40e-6; % μs per second
crystal_freq_mean = 7.37e6; % Hz

t = (0:sampling_rate:duration-sampling_rate);

clock_drift = zeros(num_nodes, length(t));
offset_change = zeros(num_nodes, length(t));

for node = 1:num_nodes
    clock_drift(node, :) = sqrt(duration) * drift_rate * randn(size(t));    
    offset_change(node, :) = linspace(0, drift_rate * duration, length(t));
end

clock_error = clock_drift + offset_change;
clock_error_seconds = clock_error / crystal_freq_mean;

figure;
plot(t, clock_error_seconds), xlabel('Timepo (segundos)'), ylabel('Error del reloj (segundos)')
grid on
title('Clock Drift')
legend('Nodo 1', 'Nodo 2', 'Nodo 3', 'Nodo 4', 'Nodo 5', 'Nodo 6', 'Nodo 7', 'Nodo 8', 'Nodo 9', 'Nodo 10')
