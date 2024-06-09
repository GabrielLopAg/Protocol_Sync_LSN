close all
clear variables

I = 7; % Numero de grados
N = 35; % Numero de nodos por grado (5, 10, 15, 20)
K = 10; % Numero de espacios en buffer por nodo
xi = 18; % Numero de ranuras de sleeping
lambda = 0.001875; % Tasa de generacion de pkts (3e-4, 3e-3, 3e-2) pkts/s
sigma = 1e-3; % seg

tau_difs = 10e-6;
tau_rts = 11e-3;
tau_cts = 11e-3;
tau_ack = 11e-3;
tau_data = 43e-3;
tau_sifs = 5e-6;

tau_msg = tau_difs + tau_rts + tau_cts + tau_data + tau_ack + 3*tau_sifs;
T = tau_msg + sigma*N; % Duración de una ranura en s

tau_data_sync = 1.6e-3;
tau_msg_sync = tau_difs + tau_data_sync + tau_sifs + tau_ack; 

tsim = 0; % medido en s

Tc = T*(xi+2); % Tiempo de ciclo
Nc = 1e5; % Ciclos que dura la simulación
Ttot = Tc*Nc; % (ranuras) Tiempo total de la simulación

t_byte = Tc; % seg

p_rel = 0.8;
p_loc = 1 - p_rel;

% Node parameters
% freq_stability = 200e-6; % -100ppm to 100ppm
% freqNode = freqNominal + (rand(N, I) - 0.5) * freqStability * freqNominal;
freq_nom = 7.3728e6; % 7.3728 MHz (KHz)
freq_desv = 40e-6;
max_offset = 40e-6; % maximum offset for initial synchronization 
clocks = zeros(N, I);
freq_loc = (randn(N, I) * freq_desv + 1 ) * freq_nom; % validar el valor de 1e-4

max_xy = [150; 50];
pos_xy = max_xy.*[rand(1,N,I) + reshape(0:I-1,[1,1,I]);
             sort(rand(1,N,I), 2)
                       ];

L = 15; % Periodo de sincronizacion

offsets = zeros(N,I);
data_offsets = [];
time_offsets = [];
data_clocks  = [];
data_freq = [];
% data_clocks = zeros(steps,N,I);
% data_offsets = data_clocks;            

contador = 0;

%%%
% clocks = clocks + T*freqNode./freqNominal + max_offset*(rand(N,I) - 0.5);
%%%

Grado = zeros(2,K,N,I); % Buffer, Nodo, Grado
buf_rel = 1;
buf_loc = 2;

% Grado(Grado>0.3) = 0;
% Grado = Grado*10/3*Ttot;
% Grado = sort(Grado,1);
% 
% npkt = numel(find(Grado));
% Grado(Grado~=0) = randperm(npkt);

id = 0;
rx_sink = [];
pkts = [];
lambda2 = lambda*N*I;
ta = 0;

t = linspace(0,tsim,contador);

% Parámetros de Evaluación
perdidos = zeros(I,1);
tiempoTx = zeros(I,1);
tiempoRx = zeros(I,1);
tiempoSp = zeros(I,1);

while tsim<Ttot
    for sync = 1:L
        for i=I:-1:1
            tsim = tsim + T;        
            clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);
            contador = contador + 1;
            offsets(:,:) = clocks - tsim;
            data_offsets(contador,:,:) = offsets;
            data_clocks(contador,:,:) = clocks;    
                data_freq(contador,:,:) = freq_loc;
            time_offsets(contador) = tsim;
            % offsets = offsets + T*freqNode./freqNominal + max_offset*(rand(N,I) - 0.5)
            
        end % ended barrido
        if sync ~= L
            tiempoSp = tiempoSp + N*T*(xi+2-I);
            tsim = tsim + T*(xi+2-I);    
            clocks = clocks + (T*(xi+2-I))*freq_loc/freq_nom + (T*(xi+2-I))*max_offset*(rand(N,I)-0.5);
            contador = contador + 1;
            offsets(:,:) = clocks - tsim;
            data_offsets(contador,:,:) = offsets;
            data_clocks(contador,:,:) = clocks;    
                data_freq(contador,:,:) = freq_loc;
            time_offsets(contador) = tsim;
        end

    end % ended sync period
    
    tsim = tsim + T;        
    clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);
    contador = contador + 1;
    offsets(:,:) = clocks - tsim;
    data_offsets(contador,:,:) = offsets;
    data_clocks(contador,:,:) = clocks;    
                data_freq(contador,:,:) = freq_loc;
    time_offsets(contador) = tsim;

    X = -(0:7)'*t_byte + tsim; % Tx 4.8KBps

    ref = 1;%randi(N);
    node = ref;
    cluster = 1;
        % Offset correction
    offset = offsets(node, cluster);
    % clocks(node, cluster) = clocks(node, cluster) - offset;    
        contador = contador + 1;
        offsets(:,:) = clocks - tsim;
        data_offsets(contador,:,:) = offsets;
        data_clocks(contador,:,:) = clocks;    
                data_freq(contador,:,:) = freq_loc;
        time_offsets(contador) = tsim;               
        % Drift correction using linear regression               
    % Y = squeeze(data_offsets(end-7:end, node, cluster));
    Y = -(0:7)'*t_byte*freq_loc(node,cluster)/freq_nom + clocks(node,cluster) - X;
    [alpha, beta] = coef(X, X+Y);
    clocks(node, cluster) = clocks(node, cluster) - beta;
        % calculate coefficients
    b = X\Y;
        % correct the local frequency of the node
    % freq_loc(node,cluster) = freq_loc(node,cluster) / (1 + b(1));      
    freq_loc(node,cluster) = freq_loc(node,cluster)/alpha;  
    data_freq(contador,:,:) = freq_loc;

    for cluster = 1:I  

        tsim = tsim + T;        
        clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);
        contador = contador + 1;
        offsets(:,:) = clocks - tsim;
        data_offsets(contador,:,:) = offsets;
        data_clocks(contador,:,:) = clocks;    
        data_freq(contador,:,:) = freq_loc; 
        time_offsets(contador) = tsim;

        % X = data_clocks(end-7:end, ref, cluster);
        X = -(0:7)'*t_byte*freq_loc(ref,cluster)/freq_nom + clocks(ref,cluster);
        
        if cluster<I
            % Cluster head sync
            node = ref;
                % Offset correction
            offset = clocks(node, cluster+1) - clocks(ref, cluster+1);
            % clocks(node, cluster+1) = clocks(node, cluster+1) - offset;   
            contador = contador + 1;
            offsets(:,:) = clocks - tsim;
            data_offsets(contador,:,:) = offsets;
            data_clocks(contador,:,:) = clocks;    
            data_freq(contador,:,:) = freq_loc;
            time_offsets(contador) = tsim;             
                % Drift correction using linear regression               
            % Y = squeeze(data_offsets(end-7:end, node, cluster+1));
            Y = -(0:7)'*t_byte*freq_loc(node,cluster+1)/freq_nom + clocks(node,cluster+1) - X;
            [alpha, beta] = coef(X, X+Y);
            clocks(node, cluster+1) = clocks(node, cluster+1) - beta;
                % calculate coefficients
            b = X\Y;
                % correct the local frequency of the node
            % freq_loc(node,cluster+1) = freq_loc(node,cluster+1) / (1 + b(1)); 
            freq_loc(node,cluster+1) = freq_loc(node,cluster+1)/alpha;
            data_freq(contador,:,:) = freq_loc;
        end

        for node = 1:N
            if node~=ref
                % Offset correction
                offset = clocks(node, cluster) - clocks(ref, cluster);
                % Drift correction using linear regression               
                % Y = squeeze(data_offsets(end-7:end, node, cluster));
                Y = -(0:7)'*t_byte*freq_loc(node,cluster)/freq_nom + clocks(node,cluster) - X;
                [alpha,beta] = coef(X, X+Y);
                clocks(node, cluster) = clocks(node, cluster) - beta;                
                % calculate coefficients
                b = X\Y;
                % correct the local frequency of the node
                % freq_loc(node,cluster) = freq_loc(node,cluster) / (1 + b(1));       
                freq_loc(node,cluster) = freq_loc(node,cluster)/alpha;
                data_freq(contador,:,:) = freq_loc;
            end
        end
    end
    tsim = tsim + T*(xi+2-2*I-1);   
    clocks = clocks + (T*(xi+2-2*I-1))*freq_loc/freq_nom + (T*(xi+2-2*I-1))*max_offset*(rand(N,I)-0.5);
    contador = contador + 1;
    offsets(:,:) = clocks - tsim;
    data_offsets(contador,:,:) = offsets;
    data_clocks(contador,:,:) = clocks;
    data_freq(contador,:,:) = freq_loc;
    time_offsets(contador) = tsim;
end % ended tsim

%% Parametro de evaluacion
figure(1)
subplot(211)
plot(time_offsets, squeeze(data_offsets(:,1,:)));legend(""+(1:I))
title('Offsets del nodo 1 en cada grado')
xlabel('Tiempo real [s]')

subplot(212)
plot(time_offsets, squeeze(data_offsets(:,:,7)));
title('Offsets del Grado 7')
xlabel('Tiempo real [s]')

ylabel('Offset local [s]')

figure(2)
plot(data_freq(:,1,1)), title('frecuencia del nodo 1 del grado 1')
xlabel('Frecuencia [Hz]')


%% Evaluación de parámetro b
[a1,b1] = coef(X,Y+X)

function [alpha,beta] = coef(t_recu, t_local)
    arguments
        t_recu (:,1) {mustBeNumeric}
        t_local (:,1) {mustBeNumeric}
    end
    N = length(t_recu);
    if N~=length(t_local)
        error("Vectors must be the same length");
    end
    alpha = (N*sum(t_recu.*t_local)-sum(t_recu)*sum(t_local)) / (N*sum(t_recu.^2)-sum(t_recu)^2);
    alpha = round(alpha,6);
    beta = (sum(t_local)-sum(t_recu)) / N;
end
