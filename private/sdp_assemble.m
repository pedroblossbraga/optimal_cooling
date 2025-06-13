function [At, b, c, CONE] = sdp_assemble(K, B, f, Pe, area)
% Assembles the matrix by vectorizing all variable dependent matrices and
% the constant one, and placing them in the columns of F
%
% INPUT
% K: Stiffness matrix for xi,theta,psi. Size mxm
% B: Cell array of length m, where every entry is the Bk matrix of size mxm
% area: Area of the domain
% Pe: Péclet number
%
% OUTPUT
% The SDP structure in SeDuMi form

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIRST LMI (the real one)
% Matrix sizes
m = size(K,1);
s = 2*m;

% Constant term (eta-eta block, bottom right)
[i,j,v] = find(K);
I = sub2ind([s s], m+i, m+j);
J = ones(size(v));
V = v;

% Terms with a (psi-psi block, top-left)
I = [I; sub2ind([s s], i, j)];
J = [J; 2*ones(size(v))];
V = [V; v];

% Terms for xi
% Recall that B{k} gives discretiazation with order psi' * B(xi) * \theta
% so rows of B{k} correspond to the psi block, columns to the theta block
for k = 1:m
    [i,j,v] = find(B{k});
    I = [I; sub2ind([s s], [i; m+j], [m+j; i])];
    J = [J; (k+3)*ones(2*size(v,1),1)];
    V = [V; (0.5*Pe)*[v; v]];
end
F1 = sparse(I, J, V, s^2, m+3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECOND LMI (Schur complement of quadratic constraint)
% This is a stupid formulation but it is simple and it avoids a
% factorization of the stiffness matrix.

% Matrix sizes
s = m+1;

% Constant term (top-left block)
[i,j,v] = find(K);
I = sub2ind([s s], i, j);
J = ones(size(v));
V = v;

% Terms with b (bottom-right entry)
I = [I; s^2];
J = [J; 3];
V = [V; 1];

% Terms for background field
for k = 1:m
    [i,~,v] = find(K(:,k));
    I = [I; sub2ind([s s], i, s*ones(size(i)))];
    I = [I; sub2ind([s s], s*ones(size(i)), i)];
    J = [J; (k+3)*ones(2*size(v,1),1)];
    V = [V; v; v];
end
F2 = sparse(I, J, V, s^2, m+3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONCATENATE THE LMIs
% Bad code, but easy; should create F using "sparse"
At = - [ F1(:,2:end); F2(:,2:end)];
c = [F1(:,1); F2(:,1)];
b = [-1; -0.25; f./area];
CONE.s = [2*m, m+1];

end
