close all

I = 7; % Numero de grados
N = 5; % Numero de nodos por grado (5, 10, 15, 20)
K = 10; % Numero de espacios en buffer por nodo
W = 16; % Numero de mini-ranuras en la ventana de contenciÃ³n (16, 32, 64, 128, 256)
xi = 18; % Numero de ranuras de sleeping
lambda = 3e-3; % Tasa de generacion de pkts (3e-4, 3e-3, 3e-2)

tsim = 0; % medido en ranuras
T = 1; % tiempo de ranura (1 ranura)
Tc = T*(xi+2); % Tiempo de ciclo
Nc = 1e5; % Ciclos que dura la simulación
Ttot = Tc*Nc; % (ranuras) Tiempo total de la simulación

Grado = zeros(K,N,I); % Buffer, Nodo, Grado
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

while tsim<Ttot*T
    for i=I:-1:1
        while ta<=tsim % Generación de pkts locales
            id = id +1;
            n = [randi(N) randi(I)];
            pos = getFreePosition(Grado(:, n(1), n(2)));
            pkts = [pkts; id n(2) ta]; % id, grado de generación, tiempo de generación
            if pos==0
                % perdidos = perdidos +1;
            else
                Grado(pos, n(1), n(2)) = id;
            end
            ta = arribo(ta, lambda2);
        end % ended generacion de pkts locales
        
        %         disp(i)
        %         aux = Grado(:,:,i);
        %         disp(aux)
        
        % Proceso de contención
        tiene_pkt = find(sum(Grado(:,:,i),1)); % Nodos que tienen pkt en buffer
        backoff = randi(W, size(tiene_pkt)); % Condicionado a size(tiene_pkt)
        ganador = find(backoff==min(backoff)); % Indice de tiene_pkt
        
        if numel(ganador) == 1
            if i>1
                %             disp("Tx " + tiene_pkt(ganador));
                pos = getFreePosition(Grado(:,tiene_pkt(ganador),i-1)); % Last free position
                if pos==0 % BUFFER LLENO
                    % perdidos = perdidos +1;
                else
                    Grado(pos,tiene_pkt(ganador),i-1) = Grado(K,tiene_pkt(ganador),i);
                end
            else % recepción en Sink
                id_r = Grado(K,tiene_pkt(ganador),1);
                rx_sink = [rx_sink id_r];
                pkts(id_r,3) = tsim-pkts(id_r,3);
            end
            Grado(:,tiene_pkt(ganador),i) = [0; Grado(1:K-1,tiene_pkt(ganador),i)];
%             tx = tx+1;
        else
            %             disp("Colision");
            Grado(:,tiene_pkt(ganador),i) = [zeros(size(ganador)); Grado(1:K-1,tiene_pkt(ganador),i)];
        end % ended contencion
        tsim = tsim + T;
    end % ended barrido
    %     disp('Nodo sink')
    tsim = tsim + T*(xi+2-I);
end % ended tsim

%% Throughput de la red
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
   ["\lambda = "+lambda; "N = "+N; "W = "+W], ...
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
   ["\lambda = "+lambda; "N = "+N; "W = "+W], ...
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
