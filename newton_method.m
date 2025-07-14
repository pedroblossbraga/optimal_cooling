function [U, V, z] = newton_method(K, B, f, Pe, M_nodes, r, tol, max_iter, N)
    tic
    % implementation of low-rank newton method for optimal cooling 
    % r (low rank!!)
    
    % --- Initialization ---
    % Problem size
    n = size(K,1);
    % Initialize U and V as eigenfunctions of K
    [U,~] = eigs(K, r, 'smallestabs');
    V = U;
    U = U ./ sqrt( sum(arrayfun(@(i) U(:,i)' * K * U(:,i), 1:r)) );
    % Initialize z as feasible
    source = zeros(n,1);
    for j = 1:n
        for i = 1:r
            source(j) = source(j) + U(:,i)' * B{j} * V(:,i);
        end
    end
    z = K \ (f + Pe*source);
    % initial lagrange multipliers
    lambda = 1;
    mu = -2*z;

    % grad descent/ascent parameters
    lr_desc=0.01;
    lr_asc=0.01;
    TOL_GRAD_METHOD=1e0;

    % % --- Initialization ---
    % U = eye(M_nodes, r);
    % V = eye(M_nodes, r);
    % % U = ones(M_nodes, r);
    % % V = ones(M_nodes, r);
    % % U = randn(M_nodes, r);
    % % V = randn(M_nodes, r);
    % 
    % z = zeros(M_nodes, 1);
    % n = size(K,1);
    % 
    % % initial lagrange multipliers
    % lambda = 1; 
    % mu = zeros(n,1);
    % % mu = ones(n,1);

    residual_history = zeros(max_iter, 1);
    residual_method = strings(max_iter, 1);  % track method: "Newton" or "Grad"

    for k = 1:max_iter

        % compute residuals
        % R = compute_residuals(K, B, f, Pe, U, V, z, lambda, mu);
        % R = -R;
        [Rz, RU, RV, Rlambda, Rmu] = compute_residuals(K, B, f, Pe, U, V, z, lambda, mu);
        
        % vector of residuals (MATLAB is column-major)
        R = [Rz; reshape([RU; RV],[],1); Rlambda; Rmu];
        R = -R;

        % check convergence
        norm_R = norm(R);
        fprintf("---- norm(R): %s\n", norm_R);
        if norm_R < tol
            fprintf('Newton converged at iter %d with ||R|| = %.2e\n', k, norm_R);
            break;
        end

        % store residual from iteration k
        residual_history(k) = norm_R;

        % check constraint and cost values
        constraint = sum(arrayfun(@(i) U(:,i)' * K * U(:,i), 1:r));
        fprintf("||U^T K U|| sum: %.4f\n", constraint);
        
        cost = z' * K * z + sum(arrayfun(@(i) V(:,i)' * K * V(:,i), 1:r));
        fprintf("Iter %d, cost: %.4e\n", k, cost);

        %% accept grad desc/asc until residuals are small enough, then use Newton
        if norm_R > TOL_GRAD_METHOD %% gradient descent/ascent
            residual_method(k) = "Grad";
            fprintf("-- (grad method) norm(R): %s\n", norm_R);

            % gradient descent for primal variables
            z = z - lr_desc * Rz;
            U = U - lr_desc * RU;
            V = V - lr_desc * RV;
            
            % gradient ascent for Lagrange Multipliers
            lambda = lambda + lr_asc * Rlambda;
            mu = mu + lr_asc * Rmu;

        else %% newton method
            residual_method(k) = "Newton";
            fprintf("-- (newton method) norm(R): %s\n", norm_R);
            
            % build jacobian matrix blocks A and Bmat (different from other B)
            % J = | A   B |
            %     | B^T 0 |
            [A, Bmat] = assemble_jacobian_blocks(K, B, Pe, U, V, lambda, mu);
            
            % solve newton system
            % [s1, s2] = solve_newton_system(A, Bmat, R);
            [s1, s2] = solve_newton_system_basic(A, Bmat, R);
            
            % extract all updates
            delta_z = s1(1:n);
            idx = n + (1 : 2*n*r);
    
            delta_UV = reshape( s1(idx), 2*n, r);
            delta_U = delta_UV(1:n,:);
            delta_V = delta_UV(n+1:2*n,:); 
            delta_lambda = s2(1);
            delta_mu = s2(2:end);
            
            %------------------------------------------------------------------
            %---------  Step size search --------------------------------------
            % alpha = 0.01;     % Initial step
            alpha=1;            % Initial step
            STEP_TOL = 1e-6;    % sigma
            BETA = 0.5;         % shrink factor
            max_ls_iter = 5;
            
            for ls_iter = 1:max_ls_iter
                % Trial update
                z_trial = z + alpha * delta_z;
                U_trial = U + alpha * delta_U;
                V_trial = V + alpha * delta_V;
                lambda_trial = lambda + alpha * delta_lambda;
                mu_trial = mu + alpha * delta_mu;
            
                % Compute residual at trial point
                R_new = compute_residuals(K, B, f, Pe, U_trial, V_trial, z_trial, lambda_trial, mu_trial);
            
                % Check Armijo condition
                if norm(R_new) < (1 - STEP_TOL * alpha) * norm_R
                    break;  % sufficient decrease
                end
            
                alpha = BETA * alpha;  % reduce step
            end
            
            % Accept update
            z = z + alpha * delta_z;
            U = U + alpha * delta_U;
            V = V + alpha * delta_V;
            lambda = lambda + alpha * delta_lambda;
            mu = mu + alpha * delta_mu;
            %------------------------------------------------------------------
        end
    end
    toc
    % plot_residual_history(residual_history, k);
    plot_residual_history(residual_history, residual_method, k, norm_R, cost, Pe, r, N);

end

% function R = compute_residuals(K, B, f, Pe, U, V, z, lambda, mu)
function [Rz, RU, RV, Rlambda, Rmu] = compute_residuals(K, B, f, Pe, U, V, z, lambda, mu)
    m = size(U, 2);
    n = size(K, 1);

    % constraint Kz = f + Pe sum_{ij} u_i^T B_j v_i e_j
    rhs = f;

    Rz = K * ( 2*z + mu );

    RU = zeros(n, m); RV = zeros(n, m);

    for i = 1:m
        RU(:,i) = 2 * lambda * K  * U(:,i);
        RV(:,i) = 2 * K * V(:,i);

        for j = 1:n
            BJ = B{j};

            % RU(:,i) = RU(:,i) - Pe * dot(mu, BJ  * V(:,i));
            % RV(:,i) = RV(:,i) - Pe * dot(mu, BJ' * U(:,i));
            RU(:,i) = RU(:,i) - Pe * mu(j) * BJ * V(:,i);
            RV(:,i) = RV(:,i) - Pe * mu(j) * BJ' * U(:,i);
        end
    end

    Rlambda = -1;
    for i = 1:m
        Rlambda = Rlambda + U(:,i)' * K * U(:,i);
    end

    source = zeros(n,1);
    for j = 1:n
        for i = 1:m
            source(j) = source(j) + U(:,i)' * B{j} * V(:,i);
        end
    end
    Rmu = K*z - f - Pe * source;

    % % vector of residuals (MATLAB is column-major)
    % R = [Rz; reshape([RU; RV],[],1); Rlambda; Rmu];
end


function [A, Bmat] = assemble_jacobian_blocks(K, B, Pe, U, V, lambda, mu)
    n = size(K,1);  % number of nodes
    r = size(U,2);  % rank

    % -------- block A --------
    % Total number of variables: z (n), U (n*r), V (n*r)
     Nvar = n + 2*n*r;

     A = sparse(Nvar, Nvar); % use sparse structure!

     % z-z block
     A(1:n, 1:n) = 2 * K;
    
     %% slow: assemble 3 arrays: i,j, value 

    ALPHA = ( -Pe * mu(1) ) * B{1};
    for j = 2:n
        % check transposes here!!!!
        ALPHA = ALPHA - ( Pe * mu(j) ) * B{j};
    end

     % U-U blocks (2 lamnbda K) and V-V blocks (2K)
     shift = n;
     for i = 1:r
         Ui_idx =  shift + (1:n);
         Vi_idx =  shift + n + (1:n);

         A(Ui_idx, Ui_idx) = (2 * lambda) * K;
         A(Vi_idx, Vi_idx) = 2 * K;
        
         % % U-V cross blocks alpha = Pe \sum mu_j B_j
         A(Ui_idx, Vi_idx) = ALPHA ; % \alpha
         A(Vi_idx, Ui_idx) = ALPHA'; % \alpha^T

         % Update shift
         shift = shift + 2*n;
     end
    
    % -------- block Bmat --------
    % size: (Nvar x 2)
    N_LagrangMult = 1+n;
    Bmat = sparse(Nvar, N_LagrangMult);

    % Bmat(1:n, 2) = K * ones(n,1); % or set Bmat(1:n,2) = K*z later if needed
    Bmat(1:n, 2:end) = K;

    % Loop over ""rank blocks"
    shift = n;
    for i = 1:r
        % row indices for Ui and Vi
        Ui_idx =  shift + (1:n);
        Vi_idx =  shift + n + (1:n);

        % first column: derivatative wrt lambda (2 K u_i \\ 0) ...
        Bmat(Ui_idx, 1) = 2 * K * U(:,i);

        % compute gammas and betas
        gamma_i_tr = zeros(n,n);
        beta_i_tr  = zeros(n,n);
        for j = 1:length(B)
            % \beta_i = - Pe [\sum_j v_i^T B_j^T \cdot e_j ]^T
            beta_i_tr(:,j) = - Pe * B{j} * V(:,i);

            % \gamma_i = - Pe \sum_j v_i^T B_j^T \cdot e_j 
            gamma_i_tr(:,j) = - Pe * B{j}' * U(:,i);
        end

        % assign gammas and betas
        Bmat(Ui_idx, 2:end) = beta_i_tr;
        Bmat(Vi_idx, 2:end) = gamma_i_tr;

        % sanity check
        assert(all(size(Bmat(Ui_idx, 2:end)) == size(beta_i_tr)));

        % Udate shift
        shift = shift + 2*n;
    end

end

function [s1, s2] = solve_newton_system_basic(A, Bmat, rhs)
    nmult = size(Bmat,2);
    M = [A, Bmat; Bmat', sparse(nmult,nmult)];

    % M_dense = full(M);
    % csvwrite('M_matrix.csv', M_dense);

    % % % regularization term
    % M = M + 1e-6 * speye(size(J));

    s = M \ rhs;
    s1 = s(1:end-nmult);
    s2 = s(end-nmult+1:end);
end

function [s1, s2] = solve_newton_system(A, Bmat, rhs)
    % solve system:
    % | A   B | = |r1|
    % | B^T 0 |   |r2|

    n1 = size(A,1);  % = n + 2nr
    %n2 = size(Bmat,2);

    % unpack R1 and R2 from rhs (R)
    % R1 = rhs(1:n1);
    % R2 = rhs(n1+1:end);

    rhs = rhs(:);
    R1 = rhs(1:n1);
    R2 = rhs(n1+1:end);


    % (LDL decomposition) A = L D L^T  # potentially change to cholesky L = chol(A, 'lower');
    [L, D, p] = ldl(A, "vector"); % sparsity-preserving LDL^T

    % t1 = B^T A^{-1} B = (L^{-1} B)^T D^{-1} (L^{-1} B)
    % Linv_B = inv(L) * Bmat;
    % t1 = Linv_B' * inv(D) * Linv_B;
    Y = L \ Bmat;         % solve L * Y = Bmat , Y = (L^{-1} B)
    Z = D \ Y;            % solve D * Z = Y , Z = D^{-1} (L^{-1} B)
    t1 = Y' * Z; % t1 = (Bˆ T A^{-1} B) = (L^{-1} B)^T D^{-1} (L^{-1} B)

    % assert(all(size(Bmat' * (A \ R1)) == size(R2)), "Incompatible sizes: Bᵗ A⁻¹ R1 and R2");
    % solve for s2
    % t1 * s2 = (B^T A^{-1} R_1) - R_2
    % s2 = t1 \ ((Bmat' * inv(A) * R1) - R2);

    BAinvR1 = Bmat' * (A \ R1);  % (2x1)
    resid2 = BAinvR1 - R2;       % now same size

    s2 = t1 \ resid2;

    % solve for s1
    s1 = inv(A) * (R1 -(Bmat * s2));
end

% function [s1, s2] = solve_newton_system(A, B, rhs)
%     n1 = size(A, 1);
%     R1 = rhs(1:n1);
%     R2 = rhs(n1+1:end);
% 
%     % LDLᵗ factorization
%     [L, D, p] = ldl(A, 'vector');  % Permutation for numerical stability
% 
%     % Solve A^{-1} B
%     Y = L \ B(p,:);     % L * Y = P * B  ⇒ Y = L⁻¹ * (P*B)
%     Z = D \ Y;          % D * Z = Y ⇒ Z = D⁻¹ Y
%     X = L' \ Z;         % Lᵗ * X = Z ⇒ X = Lᵗ⁻¹ Z = A⁻¹ B
% 
%     S = B' * X;         % Schur complement: Bᵗ A⁻¹ B
% 
%     % Solve A^{-1} R1
%     b = L \ R1(p);
%     b = D \ b;
%     b = L' \ b;
% 
%     rhs_s2 = B' * b - R2;
%     s2 = S \ rhs_s2;
% 
%     % Recover s1
%     s1 = b - X * s2;
% 
%     % Undo permutation
%     s1(p) = s1;
% end
% function plot_residual_history(residual_history, k)
%     % Trim unused entries
%     residual_history = residual_history(1:k);
% 
%     % Plot
%     figure;
%     semilogy(1:k, residual_history, 'b-o', 'LineWidth', 1.5);
%     xlabel('Iteration');
%     ylabel('||R|| (Residual Norm)');
%     title('Newton Residual Convergence');
%     grid on;
% 
%     % Save plot
%     if ~exist('data', 'dir'); mkdir('data'); end
%     saveas(gcf, 'data/newton_residual_convergence.png');
% 
%     % Optionally also save raw data
%     writematrix(residual_history, 'data/residual_history.csv');
% 
% end
function plot_residual_history(residual_history, residual_method, k, last_R, cost, Pe, rank_m, N)
    % Trim unused entries
    residual_history = residual_history(1:k);
    residual_method = residual_method(1:k);

    % Separate indices
    idx_newton = find(residual_method == "Newton");
    idx_grad   = find(residual_method == "Grad");

    % Plot
    figure;
    hold on;

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

    semilogy(idx_grad,   residual_history(idx_grad),   'ro', 'MarkerSize', 6, 'DisplayName', 'Gradient');
    semilogy(idx_newton, residual_history(idx_newton), 'bo', 'MarkerSize', 6, 'DisplayName', 'Newton');
    
    xlabel('Iteration', 'FontSize', 17);
    ylabel('||R|| (Residual Norm)', 'FontSize', 17);
    % title(sprintf('Residual Convergence (||R||=%.2e, n.iter=%d, cost=%.2e, Pe=%d, rank=%d)', ...
    %     last_R, k, cost, Pe, rank_m), ...
    %     'FontSize', 20);
    title({ ...
        sprintf('Residual history (||R||=%.2e, n.iter=%d)', last_R, k), ...
        sprintf('Cost=%.4e, Pe=%d, rank=%d, N_{mesh}=%d', cost, Pe, rank_m, N) ...
    }, 'FontSize', 20);


    legend('FontSize', 20, 'Location', 'northeast');
    grid on;
    hold off;

    % Save plot
    if ~exist('data', 'dir'); mkdir('data'); end
    saveas(gcf, 'data/newton_residual_convergence.png');

    % % Optionally also save raw data
    % T = table((1:k)', residual_history, residual_method, ...
    %           'VariableNames', {'Iteration', 'Residual', 'Method'});
    % writetable(T, 'data/residual_history.csv');
end
