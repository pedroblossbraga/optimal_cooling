%% Set up discretization of lower-bound problem for optimal cooling
% This code assumes the domain is a square

% clean up
clear
close all

% Parameters
N = 20;
tol = 1e-13;
Pe = 1;

% The heating distribution function (here, equal to 1 everywhere)
f_handle = @(x,y) ones(size(x));

% Mesh the domain
% [nodes,elements,dirichlet,area] = mesh_rectangle_symm(1, N, N);
[nodes,elements,dirichlet,area] = mesh_rectangle(1, N, N);
mesh_plot(nodes,elements,dirichlet);

% Create the discrete problem:
% M = mass matrix on the mesh
% K = stiffnes matrix on the mesh
% B = cell array with B{i}=Bi from the notes (I think - or the transpose)
% We also clean near zeros in the dataf = sdp_getF(nodes, elements, dirichlet, area, f_handle);
f = sdp_getF(nodes, elements, dirichlet, area, f_handle);
[M, K, B] = sdp_getMatrices(nodes,elements,dirichlet);
[M, K, B] = sdp_clean(M, K, B, tol);

tic
% Solve SDP if small enough
% This assumes the MATLAB path has working installations of
% * YALMIP
% * MOSEK
if N <= 25
% if N <= 250
    try % Maybe the user does not have the right packages installed...

        % Assemble the SDP in SeDuMi format (easy)
        % Here, we lift the quadratic term in the cost using an LMI. This is
        % suboptimal: we should really use a second-order cone constraint after
        % factorizing the stiffness matrix.
        [At, b, c, CONE] = sdp_assemble(K, B, f, Pe, area);

        % Call solver: data -> yalmip -> mosek
        % This is also stupid: we should call mosek directly
        y = sdpvar(size(B{1},1)+2,1);
        z = c - At*y;
        F = [];
        top = 0;
        for i = 1:length(CONE.s)
            M = reshape(z(top+1:top+CONE.s(i)^2),CONE.s(i),CONE.s(i));
            top = top + CONE.s(i)^2;
            F = [F; M>=0];
        end
        optimize(F,-b'*y)

        % Plot the optimal xi on the mesh
        figure()
        xi = zeros(size(nodes,1),1);
        [ID,~,~] = fe_locator(nodes,elements,dirichlet,1);
        xi(ID~=0) = value(y(3:end));
        trisurf(elements(:,1:3),nodes(:,1),nodes(:,2),xi,'FaceColor','interp')
        axis equal; view([0 90])

    catch
        % Something went wrong, likely the packages are not installed
        warning('Could not solve SDP. Please install YALMIP and MOSEK!')
    end
end
toc