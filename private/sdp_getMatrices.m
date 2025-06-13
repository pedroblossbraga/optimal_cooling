function [M,K,B,m] = sdp_getMatrices(nodes,elements,dirichlet)
% Calculates all M matrices
% INPUT
% nodes: Matrix of size (n° of nodes)x(2) where every row is the coordinate
% of a node
% elements: Matrix of size (n° of elements)x3 where every row are the nodes
% a triangle in counterclockwise order
% dirichlet: Matrix of size (n° of dirichlet nodes)x(2) where every row is
% a dirichlet node and its value (only zero values are supported)
% OUTPUT
% m: Number of internal nodes
% K: Stiffness matrix for xi,theta,psi. Size mxm
% Apt: Cell array of length m, where every entry is the Ak matrix of size
% mxm
% Mtt: Mass matrix for xi,theta,psi. Size mxm

% Assuming all variables are discretized with order 1
[ID,IEN,LM] = fe_locator(nodes,elements,dirichlet,1);
n_el=size(IEN,2); %Number of elements (also n_el=size(elements,1)
m = max(ID);

M = sparse(m,m);
K = sparse(m,m);
B = cell(m,1);
for k = 1:m
    B{k} = sparse(m,m);
end

nP = 9; % I should check if this is enough or too much
ord_theta = 1;
ord_psi = 1;
[xi2D, w2D] = fe_GaussLeg2DTri(nP);
[theta,gradxitheta] = fe_shapeTrivec(xi2D, ord_theta);
[psi,~] = fe_shapeTrivec(xi2D, ord_psi);

n_en = 3; % order 1
for e=1:n_el
    xnod = nodes(IEN(:,e),:);

    [eK,eApt,eM] = fe_localElems(xnod,theta,gradxitheta,psi,w2D);
   
    % VERY INEFFICIENT ASSEMBLY
    % SHOULD USE SPARSE INDEXING AND BUILD AT THE END!
    apt = cell(n_en,1);
    for k = 1:n_en
        apt{k} = spalloc(m,m,n_en^2);
    end
    mmat = spalloc(m,m,n_en^2);
    kmat = spalloc(m,m,n_en^2);
   
    for i=1:n_en
        if LM(i,e) ~= 0
            for j=1:n_en
                if LM(j,e) ~= 0
                    mmat(LM(i,e), LM(j,e)) = mmat(LM(i,e), LM(j,e)) + eM(i,j);
                    kmat(LM(i,e), LM(j,e)) = kmat(LM(i,e), LM(j,e)) + eK(i,j);
                    for k=1:n_en
                        if LM(k,e) ~= 0
                            apt{k}(LM(i,e), LM(j,e)) = apt{k}(LM(i,e), LM(j,e)) + eApt(i,j,k); 
                        end
                    end
                end
            end
        end
    end
    M = M + mmat;
    K = K + kmat;
    for k = 1:n_en
        if LM(k,e) ~= 0
            B{LM(k,e)} = B{LM(k,e)} + apt{k,1};
        end
    end
    
% The code above sets up the matrix B in the order psi'*B(xi)*theta
% Uncomment below if you need theta first...
% B = cellfun(@transpose, B, 'UniformOutput', 0);

end
end
