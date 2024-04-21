clear variables
% Parámetros de la simulación
num_nodos = 10; % Número de nodos en la red
tiempo_simulacion = 1000; % Tiempo total de simulación en segundos
dt                = 1;    % Resolución de la simulación en segundos
intervalo_muestreo = 0; % Intervalo de muestreo en segundos
frecuencia_oscilador = 32768; % Frecuencia del oscilador en Hz
desviacion_oscilador = 20; % Desviación del oscilador en ppm
frecuencia_muestreo = 0; % Frecuencia de muestreo en Hz

% Inicialización de variables
relojes_nodos = zeros(num_nodos, 1); % Inicialización de los relojes de los nodos
offset_inicial = 0; % Offset inicial de tiempo

% Bucle de simulación
for t = 1:dt:tiempo_simulacion
    % Generar offset aleatorio
    offset_aleatorio = dt * desviacion_oscilador / 1e6 * ( rand(num_nodos, 1)*2 - 1 );
    
    % Actualizar relojes de los nodos con el offset aleatorio
    relojes_nodos = relojes_nodos + dt + offset_aleatorio;
    
    % Sincronizar relojes cada cierto intervalo de muestreo 
    % con el algoritmo de FTSP
    if mod(t, frecuencia_muestreo) == 0
        % CORREGIR todo dentro del if, esto no es FTSP
        % Calcular el promedio de los relojes de los nodos
        tiempo_promedio = mean(relojes_nodos);
        
        % Calcular los offsets relativos
        offsets_relativos = relojes_nodos - tiempo_promedio;
        
        % Corregir los relojes de los nodos con los offsets relativos
        relojes_nodos = relojes_nodos - offsets_relativos;
    end
    
    % Simular la transmisión de los paquetes de sincronización cada segundo
    if mod(t, intervalo_muestreo) == 0
        
        % Envío de paquete de sincronización con el tiempo actual
        disp(['Nodo 1 envía paquete de sincronización en el tiempo ', num2str(t)]);
    end
end