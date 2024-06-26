% N = 40;
% K = 7;
% I = 7;
% xi = 18;
% lambda = 0.001875; % pkts/s
% sigma = 1e-3; % seg
% tau_difs = 10e-3;
% tau_rts = 11e-3;
% tau_cts = 11e-3;
% tau_ack = 11e-3;
% tau_data = 43e-3;
% tau_sifs = 5e-3;
% tau_msg = tau_difs + tau_rts + tau_cts + tau_data + tau_ack + 3*tau_sifs;
% T = tau_msg + sigma*N; % Duración de una ranura en s
% Tc = T*(xi+2); % Tiempo de ciclo
a = lambda * Tc;

% p_rel = 0.9;
p_loc = 1 - p_rel;

mc = zeros((K+1)^2);

p_ee = rand;
p_ee_ = 0;
p_r = 0;
pr = zeros(I,1); % p_t(i+1) [1 - p_ee(i+1)]; % i<I
pb = pr;
pt = pr;

pi_mk = zeros((K+1)^2,I);
% while abs(p_ee-p_ee_)>1e-6
%     pt(i) = (1-p_ee^N) / N/(1-p_ee);
%     pb(i) = 1-pt(i);
% 
%     p_t = pt(i);
%     p_b = pb(i);
% 
%     % (n, v)
%     % (m, u)
% 
%     m = 0;
%     n = 0;
%     u = 0;
%     v = 0;
%     mc(m*(K+1)+u+1, n*(K+1)+v+1) = (1-p_r)*(1-a); % 8
% 
%     n = 1;
%     v = 1;
%     mc(m*(K+1)+u+1, n*(K+1)+v+1) = p_r*a; % 9
% 
%     v = 0;
%     mc(m*(K+1)+u+1, n*(K+1)+v+1) = p_r*(1-a); % 10
% 
%     n = 0;
%     v = 1;
%     mc(m*(K+1)+u+1, n*(K+1)+v+1) = (1-p_r)*a; % 11
% 
%     for m = 1:K-1
%         mc(m*(K+1)+0+1, m*(K+1)+0+1)     = p_r*(1-a)*p_t + (1-p_r)*(1-a)*(1-p_t); % 12
%         mc(m*(K+1)+0+1, (m+1)*(K+1)+0+1) = p_r*(1-a)*(1-p_t); % 13
%         mc(m*(K+1)+0+1, m*(K+1)+1+1) = (1-p_r)*a*(1-p_t) + p_t*a*p_r; % 16
%         mc(m*(K+1)+0+1, (m-1)*(K+1)+1+1) = (1-p_r)*a*p_t; % 19
%         mc(m*(K+1)+0+1, (m-1)*(K+1)+0+1) = (1-p_r)*(1-a)*p_t; % 21
%         mc(m*(K+1)+0+1, (m+1)*(K+1)+1+1) = p_r*a*(1-p_t); % 22
%         mc(m*(K+1)+K+1, m*(K+1)+K+1) = (1-p_r)*(1-p_t) + p_r*p_t*p_rel; % 31
%         mc(m*(K+1)+K+1, (m+1)*(K+1)+K+1) = p_r*(1-p_t); % 32
%         mc(m*(K+1)+K+1, (m-1)*(K+1)+K+1) = (1-p_r)*p_t*p_rel; % 35
%         mc(m*(K+1)+K+1, m*(K+1)+K-1+1) = (1-p_r)*p_t*p_loc; % 36
%         mc(m*(K+1)+K+1, (m+1)*(K+1)+K-1+1) = p_r*p_t*p_loc; % 37
%     end
% 
%     for u = 1:K-1
%         mc(0*(K+1)+u+1, 0*(K+1)+u+1)   = (1-p_r)*a*p_t + (1-p_r)*(1-a)*(1-p_t); % 14
%         mc(0*(K+1)+u+1, 0*(K+1)+u+1+1) = (1-p_r)*a*(1-p_t); % 15
%         mc(0*(K+1)+u+1, 0*(K+1)+u-1+1) = (1-p_r)*(1-a)*p_t; % 17
%         mc(0*(K+1)+u+1, 1*(K+1)+u-1+1) = p_r*(1-a)*p_t; % 18
%         mc(0*(K+1)+u+1, 1*(K+1)+u+1) = p_r*(1-a)*(1-p_t) + p_r*a*p_t; % 20
%         mc(0*(K+1)+u+1, 1*(K+1)+u+1+1) = p_r*a*(1-p_t); % NO APARECE
%         mc(K*(K+1)+u+1, K*(K+1)+u+1) = (1-a)*(1-p_t) + a*p_t*p_loc; % 33
%         mc(K*(K+1)+u+1, K*(K+1)+u+1+1) = a*(1-p_t); % 34
%         mc(K*(K+1)+u+1, (K-1)*(K+1)+u+1) = (1-a)*p_t*p_rel; % 38
%         mc(K*(K+1)+u+1, (K-1)*(K+1)+u+1+1) = a*p_t*p_rel; % 39
%         mc(K*(K+1)+u+1, K*(K+1)+u-1+1) = (1-a)*p_t*p_loc; % 40
%     end
% 
%     for m = 1:K-1
%         for u = 1:K-1
%             mc(m*(K+1)+u+1, m*(K+1)+u+1) = (1-p_r)*(1-a)*(1-p_t) + p_r*(1-a)*p_t*p_rel + (1-p_r)*a*p_t*p_loc; % 23
%             mc(m*(K+1)+u+1, m*(K+1)+u+1+1) = (1-p_r)*a*(1-p_t) + p_r*a*p_t*p_rel; % 24
%             mc(m*(K+1)+u+1, m*(K+1)+u-1+1) = (1-p_r)*(1-a)*p_t*p_loc; % 25
%             mc(m*(K+1)+u+1, (m+1)*(K+1)+u+1+1) = p_r*a*(1-p_t); % 26
%             mc(m*(K+1)+u+1, (m+1)*(K+1)+u-1+1) = p_r*(1-a)*p_t*p_loc; % 27
%             mc(m*(K+1)+u+1, (m-1)*(K+1)+u+1+1) = (1-p_r)*a*p_t*p_rel; % 28
%             mc(m*(K+1)+u+1, (m+1)*(K+1)+u+1) = p_r*(1-a)*(1-p_t) + p_r*a*p_t*p_loc; % 29
%             mc(m*(K+1)+u+1, (m-1)*(K+1)+u+1) = (1-p_r)*(1-a)*p_t*p_rel; % 30
% 
%         end
%     end
% 
%     mc(K*(K+1)+0+1, (K-1)*(K+1)+0+1) = (1-a)*p_t; % 41
%     mc(K*(K+1)+0+1, (K-1)*(K+1)+1+1) = a*p_t; % 42
%     mc(K*(K+1)+0+1, K*(K+1)+1+1) = a*(1-p_t); % NO APARECE
%     mc(K*(K+1)+0+1, K*(K+1)+0+1) = (1-a)*(1-p_t); % 43
%     mc(0*(K+1)+K+1, 0*(K+1)+K+1) = (1-p_r)*(1-p_t); % 44
%     mc(0*(K+1)+K+1, 0*(K+1)+K-1+1) = (1-p_r)*p_t; % 45
%     mc(0*(K+1)+K+1, 1*(K+1)+K-1+1) = p_r*p_t; % 46
%     mc(0*(K+1)+K+1, 1*(K+1)+K+1) = p_r*(1-p_t); % 32 m=0
%     mc(K*(K+1)+K+1, (K-1)*(K+1)+K+1) = p_t*p_rel; % 47
%     mc(K*(K+1)+K+1, K*(K+1)+K-1+1) = p_t*p_loc; % 48
%     mc(K*(K+1)+K+1, K*(K+1)+K+1) = (1-p_t); % 49
% 
%     if any(abs(sum(mc, 2) - 1) > 1e-10)
%         error('The rows of the transition matrix P must sum to 1.');
%     end
% 
%     % Set up the system of linear equations
%     A = [mc' - eye((K+1)^2); ones(1, (K+1)^2)];
%     b = [zeros((K+1)^2, 1); 1];
% 
%     % Solve for the steady-state probabilities
%     pi = A\b;
%     p_ee_ = p_ee;
%     p_ee = pi(1);
% 
% end
% pt(I) = p_t;
% pb(I) = p_b;
% pr(I-1) = pt(I) *(1 - p_ee); % i<I
% p_r = pr(I-1);
% pi_mk(:,I) = pi;

for i=I:-1:1
    p_ee_ = 0;
    while abs(p_ee-p_ee_)>1e-6
    pt(i) = (1-p_ee^N) / N/(1-p_ee);
    pb(i) = 1-pt(i);
    
    p_t = pt(i);
    p_b = pb(i);
    
    % (n, v)
    % (m, u)
    
    m = 0;
    n = 0;
    u = 0;
    v = 0;
    mc(m*(K+1)+u+1, n*(K+1)+v+1) = (1-p_r)*(1-a); % 8
    
    n = 1;
    v = 1;
    mc(m*(K+1)+u+1, n*(K+1)+v+1) = p_r*a; % 9
    
    v = 0;
    mc(m*(K+1)+u+1, n*(K+1)+v+1) = p_r*(1-a); % 10
    
    n = 0;
    v = 1;
    mc(m*(K+1)+u+1, n*(K+1)+v+1) = (1-p_r)*a; % 11
    
    for m = 1:K-1
        mc(m*(K+1)+0+1, m*(K+1)+0+1)     = p_r*(1-a)*p_t + (1-p_r)*(1-a)*(1-p_t); % 12
        mc(m*(K+1)+0+1, (m+1)*(K+1)+0+1) = p_r*(1-a)*(1-p_t); % 13
        mc(m*(K+1)+0+1, m*(K+1)+1+1) = (1-p_r)*a*(1-p_t) + p_t*a*p_r; % 16
        mc(m*(K+1)+0+1, (m-1)*(K+1)+1+1) = (1-p_r)*a*p_t; % 19
        mc(m*(K+1)+0+1, (m-1)*(K+1)+0+1) = (1-p_r)*(1-a)*p_t; % 21
        mc(m*(K+1)+0+1, (m+1)*(K+1)+1+1) = p_r*a*(1-p_t); % 22
        mc(m*(K+1)+K+1, m*(K+1)+K+1) = (1-p_r)*(1-p_t) + p_r*p_t*p_rel; % 31
        mc(m*(K+1)+K+1, (m+1)*(K+1)+K+1) = p_r*(1-p_t); % 32
        mc(m*(K+1)+K+1, (m-1)*(K+1)+K+1) = (1-p_r)*p_t*p_rel; % 35
        mc(m*(K+1)+K+1, m*(K+1)+K-1+1) = (1-p_r)*p_t*p_loc; % 36
        mc(m*(K+1)+K+1, (m+1)*(K+1)+K-1+1) = p_r*p_t*p_loc; % 37
    end
    
    for u = 1:K-1
        mc(0*(K+1)+u+1, 0*(K+1)+u+1)   = (1-p_r)*a*p_t + (1-p_r)*(1-a)*(1-p_t); % 14
        mc(0*(K+1)+u+1, 0*(K+1)+u+1+1) = (1-p_r)*a*(1-p_t); % 15
        mc(0*(K+1)+u+1, 0*(K+1)+u-1+1) = (1-p_r)*(1-a)*p_t; % 17
        mc(0*(K+1)+u+1, 1*(K+1)+u-1+1) = p_r*(1-a)*p_t; % 18
        mc(0*(K+1)+u+1, 1*(K+1)+u+1) = p_r*(1-a)*(1-p_t) + p_r*a*p_t; % 20
        mc(0*(K+1)+u+1, 1*(K+1)+u+1+1) = p_r*a*(1-p_t); % NO APARECE
        mc(K*(K+1)+u+1, K*(K+1)+u+1) = (1-a)*(1-p_t) + a*p_t*p_loc; % 33
        mc(K*(K+1)+u+1, K*(K+1)+u+1+1) = a*(1-p_t); % 34
        mc(K*(K+1)+u+1, (K-1)*(K+1)+u+1) = (1-a)*p_t*p_rel; % 38
        mc(K*(K+1)+u+1, (K-1)*(K+1)+u+1+1) = a*p_t*p_rel; % 39
        mc(K*(K+1)+u+1, K*(K+1)+u-1+1) = (1-a)*p_t*p_loc; % 40
    end
    
    for m = 1:K-1
        for u = 1:K-1
            mc(m*(K+1)+u+1, m*(K+1)+u+1) = (1-p_r)*(1-a)*(1-p_t) + p_r*(1-a)*p_t*p_rel + (1-p_r)*a*p_t*p_loc; % 23
            mc(m*(K+1)+u+1, m*(K+1)+u+1+1) = (1-p_r)*a*(1-p_t) + p_r*a*p_t*p_rel; % 24
            mc(m*(K+1)+u+1, m*(K+1)+u-1+1) = (1-p_r)*(1-a)*p_t*p_loc; % 25
            mc(m*(K+1)+u+1, (m+1)*(K+1)+u+1+1) = p_r*a*(1-p_t); % 26
            mc(m*(K+1)+u+1, (m+1)*(K+1)+u-1+1) = p_r*(1-a)*p_t*p_loc; % 27
            mc(m*(K+1)+u+1, (m-1)*(K+1)+u+1+1) = (1-p_r)*a*p_t*p_rel; % 28
            mc(m*(K+1)+u+1, (m+1)*(K+1)+u+1) = p_r*(1-a)*(1-p_t) + p_r*a*p_t*p_loc; % 29
            mc(m*(K+1)+u+1, (m-1)*(K+1)+u+1) = (1-p_r)*(1-a)*p_t*p_rel; % 30
            
        end
    end
    
    mc(K*(K+1)+0+1, (K-1)*(K+1)+0+1) = (1-a)*p_t; % 41
    mc(K*(K+1)+0+1, (K-1)*(K+1)+1+1) = a*p_t; % 42
    mc(K*(K+1)+0+1, K*(K+1)+1+1) = a*(1-p_t); % NO APARECE
    mc(K*(K+1)+0+1, K*(K+1)+0+1) = (1-a)*(1-p_t); % 43
    mc(0*(K+1)+K+1, 0*(K+1)+K+1) = (1-p_r)*(1-p_t); % 44
    mc(0*(K+1)+K+1, 0*(K+1)+K-1+1) = (1-p_r)*p_t; % 45
    mc(0*(K+1)+K+1, 1*(K+1)+K-1+1) = p_r*p_t; % 46
    mc(0*(K+1)+K+1, 1*(K+1)+K+1) = p_r*(1-p_t); % 32 m=0
    mc(K*(K+1)+K+1, (K-1)*(K+1)+K+1) = p_t*p_rel; % 47
    mc(K*(K+1)+K+1, K*(K+1)+K-1+1) = p_t*p_loc; % 48
    mc(K*(K+1)+K+1, K*(K+1)+K+1) = (1-p_t); % 49
    
    if any(abs(sum(mc, 2) - 1) > 1e-10)
        error('The rows of the transition matrix P must sum to 1.');
    end
    
    % Set up the system of linear equations
    A = [mc' - eye((K+1)^2); ones(1, (K+1)^2)];
    b = [zeros((K+1)^2, 1); 1];
    
    % Solve for the steady-state probabilities
    pi = A\b;
    p_ee_ = p_ee;
    p_ee = pi(1);
    end
    if i>1
        pr(i-1) = pt(i) *(1 - p_ee); % i<I
        p_r = pr(i-1);
    end
    pi_mk(:,i) = pi;
end

pi_rel = zeros(K+1,I);
pi_loc = zeros(K+1,I);
for i = 1:I
    for k = 0:K
        for j = 0:K
            pi_loc(k+1,i) = pi_loc(k+1,i) + pi_mk(j*(K+1)+k+1,i);
            pi_rel(k+1,i) = pi_rel(k+1,i) + pi_mk(k*(K+1)+j+1,i);
        end
    end
end

%% Consumo de energía
k = 0:N-1;
W_t = 1/N./pt.*sum(k.*pi_mk(1,:)'.^k,2);
k = 1:N-1;
W_b = (1-pi_mk(1,:)')./(N*pb.*pi_mk(1,:)').*sum(k.*pi_mk(1,:)'.^k.*(N-k),2);
T_tx = (1-pi_mk(1,:)').*(p_b.*(sigma*W_b+tau_difs) + pt.*(sigma*W_t+tau_msg));

W_t = [W_t(2:I); N];
k = 0:K;
T_rx = (1-sum(pi_mk(K*(K+1)+k+1,:),1)').*(pr.*(sigma*W_t+tau_msg) + (1-pr).*(sigma*N+tau_difs+tau_rts));

T_sp = Tc-T_tx-T_rx;

P_rx = 59.9; % mW
P_tx = 52.2;
P_sp = 0;
P_tot = 1/Tc*(P_tx*T_tx + P_rx*T_rx + P_sp*T_sp);
Ptot = 1/I*sum(P_tot);

%% Throughput de la red
S1 = 1/Tc*N*pt(1)*(1-pi_mk(1,1));

%% Probabilidad de paquete perdido
m = 0:K;
u = 0:K;
p_los = zeros(I,1);
for i=1:I
    j = 1:i-1;
    p_los(i) = 1-(1-sum(pi_mk( m*(K+1)+K+1,i )))*prod(1-sum(pi_mk(K*(K+1)+u+1,j)));
end

%% Retardo
D = zeros(I,1);
k = 0:K;
Dl = Tc/a./(1-pi_loc(K,:)').*sum(k.*pi_loc',2) - Tc/2 + (xi+1)*T;
Dr = Tc./pr./(1-pi_rel(K,:)').*sum(k.*pi_rel',2) - Tc + (xi+1)*T;

for i=1:I
    D(i) = Dl(i) + sum(Dr(1:i-1));
end