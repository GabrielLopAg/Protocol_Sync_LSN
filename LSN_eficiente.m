close all
clear variables

I = 7; % Numero de grados
N = 5; % Numero de nodos por grado (5, 10, 15, 20)
K = 10; % Numero de espacios en buffer por nodo
xi = 18; % Numero de ranuras de sleeping
lambda = 3e-3; % Tasa de generacion de pkts (3e-4, 3e-3, 3e-2) pkts/s
sigma = 1e-3; % seg

tau_difs = 10e-3;
tau_rts = 11e-3;
tau_cts = 11e-3;
tau_ack = 11e-3;
tau_data = 43e-3;
tau_sifs = 5e-3;

tau_msg = tau_difs + tau_rts + tau_cts + tau_data + tau_ack + 3*tau_sifs;
T = tau_msg + sigma*N; % Duración de una ranura en s

tau_data_sync = 43e-3;
tau_msg_sync = tau_difs + tau_data_sync + tau_sifs + tau_ack; 

tsim = 0; % medido en s

Tc = T*(xi+2); % Tiempo de ciclo
Nc = 1e1; % Ciclos que dura la simulación
Ttot = Tc*Nc; % (ranuras) Tiempo total de la simulación

p_rel = 0.8;
p_loc = 1 - p_rel;

% Node parameters
% freq_stability = 200e-6; % -100ppm to 100ppm
% freqNode = freqNominal + (rand(N, I) - 0.5) * freqStability * freqNominal;
freq_nom = 7.3728e6; % 7.3728 MHz
freq_desv = 1e-4;
max_offset = 200e-6; % maximum offset for initial synchronization
clocks = zeros(N, I);
freq_loc = (randn(N, I) * freq_desv + 1 ) * freq_nom; % validar el valor de 1e-4

L = 15; % Periodo de sincronizacion

offsets = zeros(N,I);
data_offsets = [];
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
perdidos = 0;
tiempoTx = zeros(I,1);
tiempoRx = zeros(I,1);
tiempoSp = zeros(I,1);

while tsim<Ttot
    for sync = 1:L
        for i=I:-1:1
            while ta<=tsim % Generación de pkts locales
                id = id + 1;
                
                % rng("default");
                n = [randi(N) randi(I)];
                pos = getFreePosition(Grado(buf_loc, :, n(1), n(2)));                      
                
                pkts = [pkts; id n(2) ta]; % id, grado de generación, tiempo de generación
                if pos==0
                    perdidos = perdidos + 1;
                else
                    Grado(buf_loc, pos, n(1), n(2)) = id;                
                end
                ta = arribo(ta, lambda2);
            end % ended generacion de pkts locales
            
            %         disp(i)
            %         aux = Grado(:,:,i);
            %         disp(aux)
            
            % Proceso de contención
            tiene_pkt_loc = find(Grado(buf_loc,K,:,i)); % Nodos que tienen pkt en buffer local
            tiene_pkt_rel = find(Grado(buf_rel,K,:,i)); % Nodos que tienen pkt en buffer relay
            tiene_pkt = unique([tiene_pkt_loc; tiene_pkt_rel]);
    
            if i>1
                mRx = N - nnz(Grado(buf_rel, 1, :, i-1));
            else
                mRx = 0;
            end
            mTx = numel(tiene_pkt);
    
            % buscar si tiene paquetes en el búfer
            if numel(tiene_pkt)==0
                % No hay paquetes para transmitir en ese grado
                tsim = tsim + T;
                tiempo = sigma*N + tau_difs + tau_rts;
                clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);
                contador = contador + 1;
                offsets(:,:) = clocks - tsim;
                data_offsets(contador,:,:) = offsets;
                tiempoSp = tiempoSp + N*T;
                if i>1
                    tiempoSp(i-1) = tiempoSp(i-1) - mRx*tiempo;
                    tiempoRx(i-1) = tiempoRx(i-1) + mRx*tiempo;
                end
                continue
            end
    
            hn = hash(N, tsim, T);        
            
            % backoff = randi(W, size(tiene_pkt)) % Condicionado a size(tiene_pkt)
            ganador = find(hn==max(hn(tiene_pkt))); % Indice de tiene_pkt
            j = sum(hn>=ganador);
                    
            if ~ismember(ganador,tiene_pkt_rel)
                sel_buffer = buf_loc;
            elseif ~ismember(ganador,tiene_pkt_loc)
                sel_buffer = buf_rel;
            else
                sel_buffer = randsrc(1, 1, [buf_loc buf_rel; p_loc p_rel]); 
            end
            
            if i>1
                % disp("Tx " + tiene_pkt(ganador));
                pos = getFreePosition(Grado(buf_rel, :, ganador, i-1)); % Last free position
    
                if pos==0 % BUFFER RELAY LLENO
                    perdidos = perdidos +1; % ?
                    tiempo = sigma*(j-1) + tau_difs + tau_rts; % Preguntar sobre tau_rts
                    % Intenta transmitir, pero no hay Rx de CTS
                    % No aparece en las ecuaciones
                    tiempoTx(i) = tiempoTx(i) + tiempo; 
                    tiempoSp(i) = tiempoSp(i) + T - tiempo;
    
                    tiempoSp(i-1) = tiempoSp(i-1) + T;
                else
                    Grado(buf_rel, pos, ganador, i-1) = Grado(sel_buffer, K, ganador, i);
                    tiempo = sigma*(j-1) + tau_msg;
                    tiempoTx(i)   = tiempoTx(i)   + tiempo;
                    tiempoRx(i-1) = tiempoRx(i-1) + tiempo;
                    tiempoSp([i-1 i]) = tiempoSp([i-1 i]) + T - tiempo;
                end
                tiempo = sigma*(j-1) + tau_difs;
                tiempoTx(i) = tiempoTx(i) + mTx*tiempo;
                tiempoSp(i) = tiempoSp(i) + (N-1)*T - mTx*tiempo;
    
                tiempo = tiempo + tau_rts;
                tiempoRx(i-1) = tiempoRx(i-1) + mRx*tiempo;
                tiempoSp(i-1) = tiempoSp(i-1) + (N-1)*T - mRx*tiempo;
    
                tiempoSp(1:7<i-1) = tiempoSp(1:7<i-1) + N*T;
            else % recepción en Sink
                id_r = Grado(sel_buffer, K, ganador, 1);
                rx_sink = [rx_sink id_r];
                pkts(id_r,3) = tsim-pkts(id_r,3);
    
                tiempo = sigma*(j-1) + tau_msg;
                tiempoTx(i) = tiempoTx(i) + tiempo;
                tiempoSp(i) = tiempoSp(i) + T-tiempo;
                tiempo = sigma*(j-1) + tau_difs;
                tiempoTx(i) = tiempoTx(i) + mTx*tiempo;
                tiempoSp(i) = tiempoSp(i) + (N-1)*T - mTx*tiempo;
            end
            Grado(sel_buffer, :, ganador, i) = [0 Grado(sel_buffer, 1:K-1, ganador, i)];
            % tx = tx+1;
    
            tiempoSp(1:7>i) = tiempoSp(1:7>i) + N*T;
            tsim = tsim + T;        
            clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);
            contador = contador + 1;
            offsets(:,:) = clocks - tsim;
            data_offsets(contador,:,:) = offsets;
            % offsets = offsets + T*freqNode./freqNominal + max_offset*(rand(N,I) - 0.5)
            
        end % ended barrido
        %     disp('Nodo sink')
     
        % Print timestamp for each node
        % disp(['Timestamps at time ' num2str(tsim) ':']);
        % disp(relojes_nodos);
    
        if sync ~= L
            tiempoSp = tiempoSp + N*T*(xi+2-I);
            tsim = tsim + T*(xi+2-I);    
            clocks = clocks + (T*(xi+2-I))*freq_loc/freq_nom + (T*(xi+2-I))*max_offset*(rand(N,I)-0.5);
            contador = contador + 1;
            offsets(:,:) = clocks - tsim;
            data_offsets(contador,:,:) = offsets;
        end

    end % ended sync period
    
    for cluster = 1:I   
        ref = randi(N);
        if cluster>1
            tiempoTx(cluster-1) = tiempoTx(cluster-1) + tau_msg_sync;
            tiempoSp(cluster-1) = tiempoSp(cluster-1) - tau_msg_sync;

            tiempoRx(cluster-1) = tiempoRx(cluster-1) + (N-1)*tau_msg_sync;
            tiempoSp(cluster-1) = tiempoSp(cluster-1) - (N-1)*tau_msg_sync;

            X = data_offsets(end-7:end, ref, cluster-1);
        else
            X = (-7:0)'*T + tsim; % ?
        end
        tiempoRx(cluster) = tiempoRx(cluster) + tau_msg_sync;
        tiempoSp(cluster) = tiempoSp(cluster) - tau_msg_sync;

        tiempoSp = tiempoSp + N*T;
        tsim = tsim + T;
        for node = 1:N
            if node~=ref
                % Offset correction
                offset = offsets(node, cluster);
                clocks(node, cluster) = clocks(node, cluster) - offset;                
                % Drift correction using linear regression               
                Y = squeeze(data_offsets(end-7:end, node, cluster));
                % calculate coefficients
                b = X\Y;
                % correct the local frequency of the node
                freq_loc(node,cluster) = freq_loc(node,cluster) / (1 + b(1));               
            end
        end
        if cluster<I
            node = ref;
            % Offset correction
            offset = offsets(node, cluster+1);
            clocks(node, cluster+1) = clocks(node, cluster+1) - offset;                
            % Drift correction using linear regression               
            Y = squeeze(data_offsets(end-7:end, node, cluster+1));
            % calculate coefficients
            b = X\Y;
            % correct the local frequency of the node
            freq_loc(node,cluster+1) = freq_loc(node,cluster+1) / (1 + b(1));  
        end
    end
    tiempoTx(cluster) = tiempoTx(cluster) + tau_msg_sync;
    tiempoSp(cluster) = tiempoSp(cluster) - tau_msg_sync;


    tiempoRx(cluster) = tiempoRx(cluster) + (N-1)*tau_msg_sync;
    tiempoSp(cluster) = tiempoSp(cluster) - (N-1)*tau_msg_sync;

    % tiempoSp = tiempoSp + N*T;

    tiempoSp = tiempoSp + N*T*(xi+2-2*I-1);
    tsim = tsim + T*(xi+2-2*I-1);   
end % ended tsim

%% Parametro de evaluacion
% Calculo de potencia
table(tiempoSp, tiempoRx, tiempoTx, tiempoSp+tiempoRx+tiempoTx, ...
    'VariableNames',["S", "Rx", "Tx", "Suma"])
% tsim*N


% Throughput de la red
th = numel(rx_sink)/Nc

rx_sink = pkts(ismember(pkts(:,1),rx_sink),:);

% Retardo por grado
retardo = zeros(1,I);
for i = 1:I
    retardo(i) = mean( rx_sink(rx_sink(:,2)==i,3) ) / Tc;
end
figure(1);
bar(retardo);
title('Retardo promedio del paquete');
xlabel('Grado de origen');
ylabel('Retardo [ciclos]');
annotation('textbox',[0.15 0.6 0.3 0.3], 'String', ...
   ["\lambda = "+lambda; "N = "+N], ...
   'FitBoxToText', 'on');

% Paquetes perdidos
perd = zeros(1,I);
for i = 1:I
    p = numel( pkts(pkts(:,2)==i,1) );
    perd(i) = (p - numel( rx_sink(rx_sink(:,2)==i,1) )) / p;
end
figure(2);
bar(perd)
title('Probabilidad de paquete perdido');
xlabel('Grado de origen');
annotation('textbox',[0.15 0.6 0.3 0.3], 'String', ...
   ["\lambda = "+lambda; "N = "+N], ...
   'FitBoxToText', 'on');

% histogram(pkts(ismember(pkts(:,1),rx_sink),2));
% figure(2)
% histogram(pkts(:,2));

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

function hn = hash(N, tsim, T)    
    k = 0:N-1;
    p = nextprime(N); % prime number p ≥ N
    
    s = rng();
    rng(tsim/T)
    a_n = randi([1, p-1]); % PREGUNTAR a_n=0 genera el mismo ticket N veces
    b_n = randi([0, p-1]);
    
    aux = (a_n*k + b_n); 
    hn = mod(aux, p);
    rng(s);
end
