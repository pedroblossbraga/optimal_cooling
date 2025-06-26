%% Optimal Cooling via Low-Rank Newton Method
% This code assumes the domain is a unit square

% --- Clean up workspace ---
clear
close all

% --- Parameters ---
N = 20;                 % Mesh resolution
tol = 1e-13;            % Cleaning threshold
Pe = 0;                 % Péclet number (convection strength)
rank_m = 1;             % Low-rank parameter (number of u_i v_i^T terms)
max_iter = 500;          % Max Newton iterations
tol_newton = 1e-8;      % Newton convergence tolerance

% --- Heating distribution function (uniform) ---
f_handle = @(x,y) ones(size(x));

% --- Mesh the domain ---
[nodes,elements,dirichlet,area] = mesh_rectangle(1, N, N);
mesh_plot(nodes,elements,dirichlet);

% --- Assemble PDE matrices ---
f = sdp_getF(nodes, elements, dirichlet, area, f_handle);
[M, K, B] = sdp_getMatrices(nodes, elements, dirichlet);
[M, K, B] = sdp_clean(M, K, B, tol);

% --- Solve with Newton's method (low-rank formulation) ---
% [U, V, z] = newton_optimal_cooling(K, B, f, Pe, rank_m, tol_newton, max_iter);
M_nodes = size(K,1);
[U, V, z] = newton_method(K, B, f, Pe, M_nodes, rank_m, tol_newton, max_iter);

% --- Plot solution: temperature field z ---
figure()
trisurf(elements(:,1:3), nodes(:,1), nodes(:,2), z, 'FaceColor', 'interp');
title('z - Newton solution'); axis equal; view([0 90]);
colorbar

% --- Plot effective heat source Σ |u_i^T B_j v_i| ---
m = size(U, 2); n = length(B);
heat_source = zeros(n, 1);
for j = 1:n
    for i = 1:m
        heat_source(j) = heat_source(j) + abs(U(:,i)' * B{j} * V(:,i));
    end
end

xi = zeros(size(nodes,1), 1);
[ID, ~, ~] = fe_locator(nodes, elements, dirichlet, 1);
xi(ID ~= 0) = heat_source;

figure()
trisurf(elements(:,1:3), nodes(:,1), nodes(:,2), xi, 'FaceColor', 'interp')
title('|Σ u_i^T B_j v_i| (Effective Heat Source)');
axis equal; view([0 90]); colorbar
