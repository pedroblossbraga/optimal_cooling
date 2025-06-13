function [nM,nK,nB] = sdp_clean(M,K,B,tol)
% Removes matrix entries with absolute value less than or equal to tol
% INPUT
% K: Stiffness matrix for xi,theta,psi. Size mxm
% Apt: Cell array of length m, where every entry is the Ak matrix of size
% mxm
% Mtt: Mass matrix for xi,theta,psi. Size mxm
% tol: Tolerance for the removal of near zero entries
% OUTPUT
% nK: Stiffness matrix for xi,theta,psi without near zero entries. Size mxm
% nApt: Cell array of length m, where every entry is the Ak matrix of size
% mxm without near zero entries
% nMtt: Mass matrix for xi,theta,psi without near zero entries. Size mxm

% Clean K
m = size(K,1);
[i,j,v] = find(K);
mask = abs(v) >= tol;
nK = sparse(i(mask),j(mask),v(mask),m,m);

% Clean M
m = size(M,1);
[i,j,v] = find(M);
mask = abs(v) >= tol;
nM = sparse(i(mask),j(mask),v(mask),m,m);

% Clean Apt
nB = cell(size(B));
for k = 1:size(B,1)
    m = size(B{k}, 1);
    [i,j,v] = find(B{k});
    mask = abs(v) >= tol;
    nB{k} = sparse(i(mask),j(mask),v(mask),m,m);
end

end

