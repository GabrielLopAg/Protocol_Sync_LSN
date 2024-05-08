% Number of clusters and nodes per cluster
num_clusters = 4;
nodes_per_cluster = 5;
total_nodes = num_clusters * nodes_per_cluster;

% Parameters
drift = 40e-6; % drift per second for Mica2 motes (40μs/s)
max_offset = 500e-6; % maximum offset for initial synchronization
max_skew = 1e-6; % maximum skew for initial synchronization
sync_threshold = 50e-6; % synchronization threshold
sync_period = 100; % synchronization period (number of iterations)

% Initialize clocks, offsets, and skews
local_clocks = rand(total_nodes, 1) * max_offset; % local clock offsets
skews = rand(total_nodes, 1) * max_skew; % skews
offsets = zeros(total_nodes, 1); % global clock offsets

% Adjacency matrix for communication between nodes
adjacency_matrix = zeros(total_nodes, total_nodes);
for i = 1:num_clusters
    for j = 1:nodes_per_cluster
        node_id = (i - 1) * nodes_per_cluster + j;
        if j > 1
            adjacency_matrix(node_id, node_id - 1) = 1;
            adjacency_matrix(node_id - 1, node_id) = 1;
        end
        if j < nodes_per_cluster
            adjacency_matrix(node_id, node_id + 1) = 1;
            adjacency_matrix(node_id + 1, node_id) = 1;
        end
    end
end

% Simulation loop
iterations = 1000; % number of simulation iterations
for iter = 1:iterations
    % Broadcast synchronization messages
    for node_id = 1:total_nodes
        neighbors = find(adjacency_matrix(node_id, :));
        for neighbor_id = neighbors
            % Calculate difference in clocks
            delta_t = local_clocks(node_id) - local_clocks(neighbor_id);
            % Estimate offset and skew only at the end of synchronization period
            if mod(iter, sync_period) == 0
                if abs(delta_t - offsets(node_id)) < sync_threshold
                    offsets(node_id) = (offsets(node_id) + delta_t) / 2;
                    skews(node_id) = (skews(node_id) + drift) / 2;
                end
            end
        end
    end
    
    % Update local clocks
    for node_id = 1:total_nodes
        local_clocks(node_id) = local_clocks(node_id) + skews(node_id);
    end
end

% Display results
disp('Final global clock offsets:');
disp(offsets);
