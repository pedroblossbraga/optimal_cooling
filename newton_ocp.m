%% Optimal Cooling via Low-Rank Newton Method
% This code assumes the domain is a unit square

% --- Clean up workspace ---
clear
close all

% --- Parameters ---
N = 100;                 % Mesh resolution
tol = 1e-16;            % Cleaning threshold
Pe = 1;                 % Péclet number (convection strength)
rank_m = 2;             % Low-rank parameter (number of u_i v_i^T terms)
max_iter = 2000;          % Max Newton iterations
% tol_newton = 1e-8;      % Newton convergence tolerance
tol_newton = 1e-10;      % Newton convergence tolerance

% % For a Peclet > 0, we expect a cost of 3.474468954e-02
% % currently it converges to: cost: 3.4860e-02

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
M_nodes = size(K,1);
[U, V, z] = newton_method(K, B, f, Pe, M_nodes, rank_m, tol_newton, max_iter, N);

% --- Recover full temperature vector including Dirichlet nodes ---
z_full = zeros(size(nodes,1), 1);   % total number of mesh nodes
[ID, ~, ~] = fe_locator(nodes, elements, dirichlet, 1);  % ID maps interior nodes
z_full(ID ~= 0) = z;  % insert interior solution into full vector

% --- Plot solution: temperature field z ---
figure()
trisurf(elements(:,1:3), nodes(:,1), nodes(:,2), z_full, 'FaceColor', 'interp');
title('optimal z - Newton solution', ...
        'FontSize', 20);
axis equal
view([0 90]);
shading interp
colorbar

cb = colorbar;
cb.Color = 'k';           % Set colorbar text (numbers) to black
cb.Label.Color = 'k';     % Set label color (if used)
cb.FontSize = 14;         % Optional: match font size
cb.FontWeight = 'normal'; % Optional: style

ax = gca;
    set(ax, ...
        'Color', 'w', ...               % Axes background
        'XColor', 'k', 'YColor', 'k', ... % Axis label colors
        'GridColor', 'k', ...
        'MinorGridColor', 'k', ...
        'FontSize', 14, ...
        'FontWeight', 'normal');
    
    % Force black labels and title
    xlabel(ax.XLabel.String, 'Color', 'k', 'FontSize', 14);
    ylabel(ax.YLabel.String, 'Color', 'k', 'FontSize', 14);
    title(ax.Title.String, 'Color', 'k', 'FontSize', 16);
    
    % Also set figure background to white
    set(gcf, 'Color', 'w');
   
% Save plot
    if ~exist('data', 'dir'); mkdir('data'); end
    saveas(gcf, 'data/newton_optimal_z.png');

% --- Compute effective heat source Σ |u_i^T B_j v_i| ---
m = size(U, 2);  % rank
n = length(B);  % number of elements / basis functions
heat_source = zeros(n, 1);

for j = 1:n
    for i = 1:m
        heat_source(j) = heat_source(j) + abs(U(:,i)' * B{j} * V(:,i));
    end
end

% --- Interpolate heat source to full node vector (xi) ---
xi = zeros(size(nodes,1), 1);
xi(ID ~= 0) = heat_source;

% --- Plot effective heat source field ---
figure()
trisurf(elements(:,1:3), nodes(:,1), nodes(:,2), xi, 'FaceColor', 'interp');
title('|Σ u_i^T B_j v_i| (Effective Heat Source)', ...
        'FontSize', 20);
axis equal
view([0 90]);
shading interp
colorbar

cb = colorbar;
cb.Color = 'k';           % Set colorbar text (numbers) to black
cb.Label.Color = 'k';     % Set label color (if used)
cb.FontSize = 14;         % Optional: match font size
cb.FontWeight = 'normal'; % Optional: style

ax = gca;
    set(ax, ...
        'Color', 'w', ...               % Axes background
        'XColor', 'k', 'YColor', 'k', ... % Axis label colors
        'GridColor', 'k', ...
        'MinorGridColor', 'k', ...
        'FontSize', 14, ...
        'FontWeight', 'normal');
    
    % Force black labels and title
    xlabel(ax.XLabel.String, 'Color', 'k', 'FontSize', 14);
    ylabel(ax.YLabel.String, 'Color', 'k', 'FontSize', 14);
    title(ax.Title.String, 'Color', 'k', 'FontSize', 16);
    
    % Also set figure background to white
    set(gcf, 'Color', 'w');

% Save plot
if ~exist('data', 'dir'); mkdir('data'); end
saveas(gcf, 'data/newton_effective_heat_source.png');
