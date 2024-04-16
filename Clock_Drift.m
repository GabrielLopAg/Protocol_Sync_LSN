clear all
% Initial parameters
numNodes = 3;
timeSim = 50; 
freqStability = 200e-6; % -100ppm to 100ppm
freqNominal = 7.3728e6; % 7.3728 MHz
timeStep = 1; % 1 second time step

% Initialize frequency of each node
freqNode = freqNominal + (rand(numNodes, 1) - 0.5) * freqStability * freqNominal;

% Initialize time of each node
timeNode = zeros(numNodes, 1);

% Initialize time synchronization error
timeSyncError = zeros(timeSim, 1);

% Reference node
refNode = 1;

% Simulation
for t = 1:timeStep:timeSim
    % Update time of each node
    timeNode = timeNode + timeStep * freqNode / freqNominal;
    
    % Time synchronization error between two motes
    timeSyncError(t) = abs(timeNode(2) - timeNode(3)); 
end

% Plot time synchronization error
figure;
plot((1:timeStep:timeSim), timeSyncError);
xlabel('Time');
ylabel('Time synchronization error');
title('Time synchronization error between two motes');
grid on;