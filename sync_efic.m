close all
clear variables

I = 7; % Numero de grados
N = 3; % Numero de nodos por grado (5, 10, 15, 20)
K = 10; % Numero de espacios en buffer por nodo
xi = 18; % Numero de ranuras de sleeping
lambda = 0.001875e3; % Tasa de generacion de pkts (3e-4, 3e-3, 3e-2) pkts/s
sigma = 1e-3; % seg

tau_difs = 10e-6;
tau_rts = 11e-3;
tau_cts = 11e-3;
tau_ack = 11e-3;
tau_data = 43e-3;
tau_sifs = 5e-6;

tau_msg = tau_difs + tau_rts + tau_cts + tau_data + tau_ack + 3*tau_sifs;
T = tau_msg + sigma*N; % Duración de una ranura en s

tau_data_sync = 4e-3;
tau_msg_sync = tau_difs + tau_data_sync + tau_sifs + tau_ack; 

tsim = 0; % medido en s

Tc = T*(xi+2); % Tiempo de ciclo
Nc = 1e5; % Ciclos que dura la simulación
Ttot = Tc*Nc; % (ranuras) Tiempo total de la simulación

p_rel = 0.8;
p_loc = 1 - p_rel;

% Node parameters
% freq_stability = 200e-6; % -100ppm to 100ppm
% freqNode = freqNominal + (rand(N, I) - 0.5) * freqStability * freqNominal;
freq_nom = 7.3728e6; % 7.3728 MHz
freq_desv = 40e-6;
max_offset = 20e-6; % maximum offset for initial synchronization
clocks = zeros(N, I);
freq_loc = (randn(N, I) * freq_desv + 1 ) * freq_nom; % validar el valor de 1e-4

max_xy = [150; 50];
pos_xy = max_xy.*[rand(1,N,I) + reshape(0:I-1,[1,1,I]);
             sort(rand(1,N,I), 2)
                       ];

L = 10; % Periodo de sincronizacion

offsets = zeros(N,I);
data_offsets = [];
time_offsets = [];
data_clocks  = [];
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
            time_offsets(contador) = tsim;
        end

    end % ended sync period
    
    tsim = tsim + T;        
    clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);
    contador = contador + 1;
    offsets(:,:) = clocks - tsim;
    data_offsets(contador,:,:) = offsets;
    data_clocks(contador,:,:) = clocks;
    time_offsets(contador) = tsim;

    X = (0:7)'*1/4.8e3 + tsim; % Tx 4.8KBps

    ref = randi(N);
    node = ref;
    cluster = 1;
        % Offset correction
    offset = offsets(node, cluster);
    clocks(node, cluster) = clocks(node, cluster) - offset;    
        contador = contador + 1;
        offsets(:,:) = clocks - tsim;
        data_offsets(contador,:,:) = offsets;
        data_clocks(contador,:,:) = clocks;
        time_offsets(contador) = tsim;               
        % Drift correction using linear regression               
    % Y = squeeze(data_offsets(end-7:end, node, cluster));
    Y = (0:7)'*1/4.8e3*freq_loc(node,cluster)/freq_nom;
        % calculate coefficients
    b = X\Y;
        % correct the local frequency of the node
    freq_loc(node,cluster) = freq_loc(node,cluster) / (1 + b(1));  

    for cluster = 1:I  

        tsim = tsim + T;        
        clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);
        contador = contador + 1;
        offsets(:,:) = clocks - tsim;
        data_offsets(contador,:,:) = offsets;
        data_clocks(contador,:,:) = clocks;
        time_offsets(contador) = tsim;

        % X = data_clocks(end-7:end, ref, cluster);
        X = clocks(ref,cluster) + (0:7)'*1/4.8e3*freq_loc(ref,cluster)/freq_nom;
        
        if cluster<I
            % Cluster head sync
            node = ref;
                % Offset correction
            offset = offsets(node, cluster+1);
            clocks(node, cluster+1) = clocks(node, cluster+1) - offset;   
            contador = contador + 1;
            offsets(:,:) = clocks - tsim;
            data_offsets(contador,:,:) = offsets;
            data_clocks(contador,:,:) = clocks;
            time_offsets(contador) = tsim;             
                % Drift correction using linear regression               
            % Y = squeeze(data_offsets(end-7:end, node, cluster+1));
            Y = (1:8)'*1/4.8e3*freq_loc(node,cluster)/freq_nom + clocks(node,cluster) - tsim;
                % calculate coefficients
            b = X\Y;
                % correct the local frequency of the node
            freq_loc(node,cluster+1) = freq_loc(node,cluster+1) / (1 + b(1));  
        end

        for node = 1:N
            if node~=ref
                % Offset correction
                offset = offsets(node, cluster);
                clocks(node, cluster) = clocks(node, cluster) - offset;                
                % Drift correction using linear regression               
                % Y = squeeze(data_offsets(end-7:end, node, cluster));
                Y = (1:8)'*1/4.8e3*freq_loc(node,cluster)/freq_nom + clocks(node,cluster) - tsim;
                % calculate coefficients
                b = X\Y;
                % correct the local frequency of the node
                freq_loc(node,cluster) = freq_loc(node,cluster) / (1 + b(1));               
            end
        end
    end
    tsim = tsim + T*(xi+2-2*I-1);   
    clocks = clocks + (T*(xi+2-I))*freq_loc/freq_nom + (T*(xi+2-I))*max_offset*(rand(N,I)-0.5);
    contador = contador + 1;
    offsets(:,:) = clocks - tsim;
    data_offsets(contador,:,:) = offsets;
    data_clocks(contador,:,:) = clocks;
    time_offsets(contador) = tsim;
end % ended tsim

%% Parametro de evaluacion
plot(time_offsets, squeeze(data_offsets(:,2,:)));legend(""+(1:I))
