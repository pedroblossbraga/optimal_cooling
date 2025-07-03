function [U, V, z] = newton_method(K, B, f, Pe, M_nodes, r, tol, max_iter)
    % implementation of low-rank newton method for optimal cooling 
    % r (low rank!!)
    
    % --- Initialization ---
    U = eye(M_nodes, r);
    V = eye(M_nodes, r);
    z = zeros(M_nodes, 1);
    n = size(K,1);

    % initial lagrange multipliers
    lambda = 1; 
    mu = zeros(n,1);

    for k = 1:max_iter

        % compute residuals
        R = compute_residuals(K, B, f, Pe, U, V, z, lambda, mu);
        
        % check convergence
        norm_R = norm(R);
        fprintf("---- norm(R): %s\n", norm_R);
        if norm_R < tol
            fprintf('Newton converged at iter %d with ||R|| = %.2e\n', k, norm_R);
            break;
        end
        
        constraint = sum(arrayfun(@(i) U(:,i)' * K * U(:,i), 1:r));
        fprintf("||U^T K U|| sum: %.4f\n", constraint);
        
        cost = z' * K * z + sum(arrayfun(@(i) V(:,i)' * K * V(:,i), 1:r));
        fprintf("Iter %d, cost: %.4e\n", k, cost);


        % %% check temperature field instead of all residuals
        % norm_Rz = norm(Rz);
        % fprintf("---- norm(Rz): %s\n", norm_Rz);
        % if norm_Rz < tol
        %     fprintf('Newton converged at iter %d with ||Rz|| = %.2e\n', k, norm_Rz);
        %     break;
        % end
        
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
        delta_UV = reshape( s1(idx), 2*n, r );
        delta_U = delta_UV(1:n,:);
        delta_V = delta_UV(n+1:2*n,:);
        delta_lambda = s2(1);
        delta_mu = s2(2:end);
        
        % % step size search
        % alpha = 0.01; % initial step
        % % alpha = 1;
        % z_new = z + alpha * delta_z;
        %     U_new = U + alpha * delta_U;
        %     V_new = V + alpha * delta_V;
        %     lambda_new = lambda + alpha * delta_lambda;
        %     mu_new = mu + alpha * delta_mu;

        %------------------------------------------------------------------
        %---------  Step size search --------------------------------------
        % alpha = 0.01;     % Initial step
        alpha=1;            % Initial step
        STEP_TOL = 1e-4;    % sigma
        BETA = 0.5;         % shrink factor
        max_ls_iter = 200;
        
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

function R = compute_residuals(K, B, f, Pe, U, V, z, lambda, mu)
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
    
    % Rmu = K*z - f;
    % for i = 1:m
    %     for j = 1:n
    %         BJ = B{j};
    %         Rmu = Rmu - Pe * U(:,i)' * BJ * V(:,i);
    %     end
    % end
    %$ subtracting scalar f from vector Rmu
    source = zeros(n,1);
    for j = 1:n
        for i = 1:m
            source(j) = source(j) + U(:,i)' * B{j} * V(:,i);
        end
    end
    Rmu = K*z - f - Pe * source;

    % vector of residuals (MATLAB is column-major)
    R = -[Rz; reshape([RU; RV],[],1); Rlambda; Rmu];
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
         % cross = sparse(n,n);
         % for j = 1:length(B)
         %     cross = cross - Pe * mu * B{j};
         % end
           
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
    % Bmat(1:n, 1:n) = 0;
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
            % gradU = gradU - Pe * B{j} * V(:,i);
            beta_i_tr(:,j) = - Pe * B{j} * V(:,i);

            % \gamma_i = - Pe \sum_j v_i^T B_j^T \cdot e_j 
            % gradV = gradV - Pe * B{j}' * U(:,i);
            gamma_i_tr(:,j) = - Pe * B{j}' * U(:,i);
        end

        % assign gammas and betas
        Bmat(Ui_idx, 2:end) = beta_i_tr;
        Bmat(Vi_idx, 2:end) = gamma_i_tr;

        % Udate shift
        shift = shift + 2*n;
    end

end

function [s1, s2] = solve_newton_system_basic(A, Bmat, rhs)
    nmult = size(Bmat,2);
    M = [A, Bmat; Bmat', sparse(nmult,nmult)];
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