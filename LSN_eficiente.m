close all
clear variables

global Grado K buf_loc buf_rel tiempoTx tiempoRx tiempoSp tau_difs tau_rts tau_msg sigma T perdidos th n_pkt retardos pkts tsim rx_sink ;
global N I Ttot Tc Nc tiempo t_byte xi std freq_loc freq_nom clocks max_offset offsets contador data_clocks data_offsets data_freq time_offsets tau_msg_sync;

% Initialization Parameters
I = 7; % Number of degrees
N = 35; % Number of nodes per degree (5, 10, 15, 20)
p = nextprime(N);
K = 7; % Number of buffer spaces per node
xi = 18; % Number of sleeping slots
lambda = 0.001875; % Packet generation rate (3e-4, 3e-3, 3e-2) pkts/s
sigma = 1e-3; % seconds

% Define time constants
tau_difs = 10e-3;
tau_rts = 11e-3;
tau_cts = 11e-3;
tau_ack = 11e-3;
tau_data = 43e-3;
tau_sifs = 5e-3;

tau_msg = tau_difs + tau_rts + tau_cts + tau_data + tau_ack + 3*tau_sifs;
T = tau_msg + sigma*N; % Duración de una ranura en seg

tau_data_sync = 40e-3;
tau_msg_sync = tau_difs + tau_data_sync + tau_sifs + tau_ack; 

% Simulation Parameters
tsim = 0; % medido en seconds
contador = 0;
tiempo = 0;
Nc = 1e2; % Ciclos que dura la simulación
Tc = T * (xi + 2); % Tiempo de ciclo
Ttot = Tc * Nc; % (ranuras) Tiempo total de la simulación
L = 20; % Periodo de Sync
ta = L * Tc;
t_byte = Tc; % seg
buf_rel = 1;
buf_loc = 2;
p_rel = 0.8;
p_loc = 1 - p_rel;
id = 0;
lambda2 = lambda * N * I;

% Node parameters
freq_nom = 7.3728e6; % 7.3728 MHz
freq_desv = 1e-6;
clocks = zeros(N, I);
freq_loc = (randn(N, I) * freq_desv + 1) * freq_nom;
max_offset = 0; % maximum offset for initial synchronization
mu = 0;
std = 1e-6;

% Inicialización de variables
Grado = zeros(2,K,N,I); % Buffer, Nodo, Grado
offsets = zeros(N,I);
rx_sink = [];
pkts = [];
data_offsets = [];
time_offsets = [];
data_clocks  = [];
data_freq = [];
buf_rel = 1;
buf_loc = 2;

max_xy = [150; 50];
pos_xy = max_xy.*[rand(1,N,I) + reshape(0:I-1,[1,1,I]);
             sort(rand(1,N,I), 2)
                       ];

% Parámetros de Evaluación
th = zeros(I,1);
n_pkt = zeros(I,1);
retardos = zeros(I,1);
perdidos = zeros(I,1);
tiempoTx = zeros(I,1);
tiempoRx = zeros(I,1);
tiempoSp = zeros(I,1);

while tsim < Ttot
    tiempo = N*sigma + tau_difs + tau_rts;
    tiempoSp(I) = tiempoSp(I) - N*tiempo;
    tiempoRx(I) = tiempoRx(I) + N*tiempo;
    for sync = 1:L
        for i = I:-1:1            
            while ta <= tsim % Generación de pkts locales
                id = id + 1;
                n = [randi(N), randi(I)];
                pos = getFreePosition(Grado(buf_loc, :, n(1), n(2)));
                n_pkt(n(2)) = n_pkt(n(2)) + 1;

                % pkts = [pkts; id n(2) ta]; % id, grado de generación, tiempo de generación
                if pos == 0
                    perdidos(n(2)) = perdidos(n(2)) + 1;
                else
                    Grado(buf_loc, pos, n(1), n(2)) = id;
                    pkts = [pkts; id n(2) ta]; % id, grado de generación, tiempo de generación
                end
                ta = arribo(ta, lambda2);
            end

            % Proceso de contención
            tiene_pkt_loc = find(Grado(buf_loc, K, :, i)); % Nodos que tienen pkt en buffer local
            tiene_pkt_rel = find(Grado(buf_rel, K, :, i)); % Nodos que tienen pkt en buffer relay
            tiene_pkt = unique([tiene_pkt_loc; tiene_pkt_rel]);

            if i > 1
                mRx = N - nnz(Grado(buf_rel, 1, :, i-1));
            else
                mRx = 0;
            end
            mTx = numel(tiene_pkt);

            if isempty(tiene_pkt) % Buscar si tiene paquetes en el búfer
                % No hay paquetes para transmitir en ese grado
                timeDuration = T;
                updateSimulationTime(timeDuration);
                tiempo = tau_difs + sigma*N + tau_rts + tau_sifs;
                tiempoSp = tiempoSp + N*T;
                if i>1
                    tiempoSp(i-1) = tiempoSp(i-1) - mRx*tiempo;
                    tiempoRx(i-1) = tiempoRx(i-1) + mRx*tiempo;
                end
                continue;
            end
            
            hn = hash(p, N, tsim, T); 
            ganador = find(hn==max(hn(tiene_pkt))); % Indice de tiene_pkt
            j = sum(hn>=hn(ganador));

            if ~ismember(ganador, tiene_pkt_rel)
                sel_buffer = buf_loc;
            elseif ~ismember(ganador, tiene_pkt_loc)
                sel_buffer = buf_rel;
            else
                sel_buffer = randsrc(1, 1, [buf_loc, buf_rel; p_loc, p_rel]);
            end

            % Proceso de transmisión
            processTransmission(ganador, sel_buffer, i, j, mRx, mTx);
            tiempoSp(1:7>i) = tiempoSp(1:7>i) + N * T; 
            timeDuration = T;
            updateSimulationTime(timeDuration);
        end % ended barrido

        if sync ~= L
            % Sleeping time
            tiempoSp = tiempoSp + N * T * (xi + 2 - I);
            timeDuration = T * (xi + 2 - I);
            updateSimulationTime(timeDuration);
        end
    end % ended sync period

    tiempoSp = tiempoSp + N * T;   
    timeDuration = T;
    updateSimulationTime(timeDuration);

    % Synchronization
    syncProtocol();

    tiempoSp = tiempoSp + N * T * (xi + 2 - 2 * I - 1);
    timeDuration = T * (xi + 2 - 2 * I - 1);
    updateSimulationTime(timeDuration);    
end % end tsim

%% Parametro de evaluacion
% Calculo de potencia
table(tiempoSp, tiempoRx, tiempoTx, tiempoSp+tiempoRx+tiempoTx, ...
    'VariableNames',["S", "Rx", "Tx", "Suma"])
% tsim*N
P_rx = 59.9; % mW
P_tx = 52.2; % mW
P_sp = 0;    % mW
% Potencia promedio consumida por nodo [mW]
P_prom = (sum(tiempoRx)*P_rx + sum(tiempoTx)*P_tx + sum(tiempoSp)*P_sp) / N/tsim/I

%% Throughput de la red (pkts/s)
n = th;
th = sum(th)/tsim; % pkts/seg

% Retardo por grado
% retardo = zeros(1,I);
% for i = 1:I
%     retardo(i) = mean( rx_sink(rx_sink(:,2)==i,3) ) / Tc;
% end
retardos = retardos./n;
figure(1);
bar(retardos);
title('Retardo promedio del paquete');
xlabel('Grado de origen');
ylabel('Retardo [s]');
annotation('textbox',[0.15 0.6 0.3 0.3], 'String', ...
   ["\lambda = "+lambda; "N = "+N], ...
   'FitBoxToText', 'on');

% Paquetes perdidos
% perd = zeros(1,I);
% for i = 1:I
%     p = numel( pkts(pkts(:,2)==i,1) );
%     perd(i) = (p - numel( rx_sink(rx_sink(:,2)==i,1) )) / p;
% end
perdidos = perdidos./n_pkt;
figure(2);
bar(perdidos)
title('Probabilidad de paquete perdido');
xlabel('Grado de origen');
annotation('textbox',[0.15 0.6 0.3 0.3], 'String', ...
   ["\lambda = "+lambda; "N = "+N], ...
   'FitBoxToText', 'on');

% histogram(pkts(ismember(pkts(:,1),rx_sink),2));
% figure(2)
% histogram(pkts(:,2));

%% Graficas de offset
figure(3)
plot(time_offsets, data_offsets(:,:,1)), grid on, title('Offsets de los nodos de un grado'), %xlim([0 300])
xlabel('Tiempo (s)'), ylabel('Magnitud del offset (s)')

%% Funciones 
function updateSimulationTime(timeDuration)
    global tsim N I std freq_loc freq_nom clocks max_offset contador data_clocks offsets data_offsets data_freq time_offsets;    

    tsim = tsim + timeDuration;
    contador = contador + 1;
    freq_loc = (timeDuration * randn(N, I) * std + 1) .* freq_loc;
    clocks = clocks + timeDuration * freq_loc / freq_nom + timeDuration * max_offset * (rand(N, I) - 0.5);
    offsets(:,:) = clocks - tsim;
    data_clocks(contador, :, :) = clocks;
    data_offsets(contador, :, :) = offsets;    
    data_freq(contador, :, :) = freq_loc;
    time_offsets(contador) = tsim;
end

function processTransmission(ganador, sel_buffer, i, j, mRx, mTx)
    global tsim Grado N K buf_rel tiempo tiempoTx tiempoRx tiempoSp tau_difs tau_rts tau_msg sigma T perdidos retardos th pkts rx_sink;

    if i > 1
        pos = getFreePosition(Grado(buf_rel, :, ganador, i-1)); % Last free position
        if pos == 0 % BUFFER RELAY LLENO
            aux = pkts(Grado(sel_buffer, K, ganador, i)==pkts(:,1),2);
            perdidos(aux) = perdidos(aux) +1; % ?
            pkts(Grado(sel_buffer, K, ganador, i)==pkts(:,1),:) = [];
            tiempo = sigma * (j - 1) + tau_difs + tau_rts; % Preguntar sobre tau_rts
            % Intenta transmitir, pero no hay Rx de CTS
            % No aparece en las ecuaciones
            tiempoTx(i) = tiempoTx(i) + tiempo;
            tiempoSp(i) = tiempoSp(i) + T - tiempo;
            tiempoSp(i-1) = tiempoSp(i-1) + T;
        else
            Grado(buf_rel, pos, ganador, i-1) = Grado(sel_buffer, K, ganador, i);
            tiempo = sigma * (j - 1) + tau_msg;
            tiempoTx(i) = tiempoTx(i) + tiempo;
            tiempoRx(i-1) = tiempoRx(i-1) + tiempo;
            tiempoSp([i-1, i]) = tiempoSp([i-1, i]) + T - tiempo;
        end
        tiempo = sigma * (j - 1) + tau_difs;
        tiempoTx(i) = tiempoTx(i) + (mTx-1) * tiempo;
        tiempoSp(i) = tiempoSp(i) + (N - 1) * T - (mTx-1) * tiempo;

        tiempo = N*sigma + tau_difs + tau_rts;
        tiempoRx(i-1) = tiempoRx(i-1) + (mRx-1) * tiempo;
        tiempoSp(i-1) = tiempoSp(i-1) + (N - 1) * T - (mRx-1) * tiempo;
        tiempoSp(1:7 < i-1) = tiempoSp(1:7 < i-1) + N * T;
    else % recepción en Sink
        % id_r = Grado(sel_buffer, K, ganador, 1);
        % rx_sink = [rx_sink, id_r];        
        % pkts(id_r, 3) = tsim - pkts(id_r, 3);
        aux = pkts(Grado(sel_buffer, K, ganador, i)==pkts(:,1),[2 3]);
        th(aux(1)) = th(aux(1)) + 1;
        retardos(aux(1)) = retardos(aux(1)) + tsim - aux(2);
        pkts(Grado(sel_buffer, K, ganador, i)==pkts(:,1), :) = [];

        tiempo = sigma * (j - 1) + tau_msg;
        tiempoTx(i) = tiempoTx(i) + tiempo;
        tiempoSp(i) = tiempoSp(i) + T - tiempo;
        tiempo = sigma * (j - 1) + tau_difs;
        tiempoTx(i) = tiempoTx(i) + (mTx-1) * tiempo;
        tiempoSp(i) = tiempoSp(i) + (N - 1) * T - (mTx-1) * tiempo;
    end
    Grado(sel_buffer, :, ganador, i) = [0, Grado(sel_buffer, 1:K-1, ganador, i)];
end

function syncProtocol()
    global tsim T N I t_byte freq_loc freq_nom clocks contador data_freq tiempoSp tiempoRx tiempoTx tau_msg_sync;                            
       
    % Correción del primer nodo de referencia
    ref = randi(N);
    node = ref;
    cluster = 1;

    X = -(0:7)' * t_byte + tsim; % Tx 4.8KBps | t_byte = Tc
    Y = -(0:7)' * t_byte * freq_loc(node, cluster)/freq_nom + clocks(node, cluster) - X;
    [alpha, beta] = coef(X, X+Y);
    clocks(node, cluster) = (clocks(node, cluster) - beta)/alpha;
    freq_loc(node, cluster) = freq_loc(node, cluster)/alpha;  
    data_freq(contador,:,:) = freq_loc;

    % Propagación de la sync a toda la red
    for cluster = 1:I
        % Cluster Head Tx
        tiempoTx(cluster) = tiempoTx(cluster) + tau_msg_sync;
        tiempoSp(cluster) = tiempoSp(cluster) - tau_msg_sync;

        % Cluster Rx
        tiempoRx(cluster) = tiempoRx(cluster) + (N-1)*tau_msg_sync;
        tiempoSp(cluster) = tiempoSp(cluster) - (N-1)*tau_msg_sync;

        tiempoSp = tiempoSp + N*T;
        timeDuration = T;
        updateSimulationTime(timeDuration);

        X = -(0:7)' * t_byte * freq_loc(ref, cluster)/freq_nom + clocks(ref, cluster);
        
        if cluster<I
            % Cluster head sync
            node = ref;                                        
                                      
            Y = -(0:7)' * t_byte * freq_loc(node,cluster+1)/freq_nom + clocks(node, cluster+1) - X;
            [alpha, beta] = coef(X, X+Y);
            
            clocks(node, cluster+1) = (clocks(node, cluster+1) - beta)/alpha; % Offset correction             
            freq_loc(node, cluster+1) = freq_loc(node, cluster+1)/alpha; % Drift correction using linear regression 
            data_freq(contador,:,:) = freq_loc;
        end

        for node = 1:N
            if node~=ref                                
                % Drift correction using linear regression                               
                Y = -(0:7)' * t_byte * freq_loc(node, cluster)/freq_nom + clocks(node, cluster) - X;
                [alpha, beta] = coef(X, X+Y);
                clocks(node, cluster) = (clocks(node, cluster) - beta)/alpha;
                freq_loc(node, cluster) = freq_loc(node, cluster)/alpha;
                data_freq(contador,:,:) = freq_loc;
            end
        end % end intra-grado sync
    end % end inter-grado sync
end

function ta = arribo(ti, lambda)
    u = (1e6*rand)/1e6;
    nuevot = -(1/lambda)*log(1-u);
    ta = ti+nuevot;
end

function pos = getFreePosition(v)
    pos = find(v,1) - 1;
    if isempty(pos)
        pos = numel(v);
    end
end

function hn = hash(p, N, tsim, T)    
    k = 0:N-1;
    % p = nextprime(N); % prime number p ≥ N
    
    s = rng();
    rng(tsim/T)
    a_n = randi([1, p-1]); % PREGUNTAR a_n=0 genera el mismo ticket N veces
    b_n = randi([0, p-1]);
    
    aux = (a_n*k + b_n); 
    hn = mod(aux, p);
    rng(s);
end

function [alpha,beta] = coef(t_recu, t_local)
    arguments
        t_recu (:,1) {mustBeNumeric}
        t_local (:,1) {mustBeNumeric}
    end
    N = length(t_recu);
    if N~=length(t_local)
        error("Vectors must be the same length");
    end
    alpha = (N * sum(t_recu.*t_local) - sum(t_recu) * sum(t_local)) / (N * sum(t_recu.^2) - sum(t_recu)^2);
    % alpha = round(alpha,6);
    beta = mean(t_local)-alpha*mean(t_recu);
end