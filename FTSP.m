clear all
N = 5; % number of nodes for each cluster
I = 7; % number of clusters
clocks = zeros(N,I); % clocks of the nodes of the entire network
max_offset = 200e-6;
freq_nom = 7.3728e6;
T = 0.121; 
freq_loc = (randn(size(clocks)) * 1e-4 + 1 ) * freq_nom;
steps = 200;
root_time = 0; % root node time

offsets = zeros(N,I);
data_clocks = zeros(steps,N,I);
data_offsets = data_clocks;

for i = 1:T:steps
    clocks = clocks + T*freq_loc/freq_nom + T*max_offset*(rand(N,I)-0.5);    
    offsets(:,:) = clocks - i*T;
    data_offsets(i,:,:) = offsets;
    
    if i==50 || i==100 %|| i==900      
        for cluster = 1:I            
            X = (i-7:i)';
            for node = 1:N
                % Offset correction
                offset = offsets(node, cluster);
                clocks(node, cluster) = clocks(node, cluster) - offset;                
                % Drift correction using linear regression               
                Y = squeeze(data_offsets(i-7:i,node,cluster));
                % calculate coefficients
                b = X\Y;
                % correct the local frequency of the node
                freq_loc(node,cluster) = freq_loc(node,cluster) / (1 + b(1));               
            end
        end
    end % end if
        
    root_time = root_time + T;
end
freq_loc
clocks
plot(data_offsets(:,:,1)), grid on, title('offset')

            