function F = sdp_getF(nodes,elements,dirichlet,area,f)
% Assembles the F vector
% INPUT 
% nodes: Matrix of size (n° of nodes)x(2) where every row is the coordinate
% of a node
% elements: Matrix of size (n° of elements)x3 where every row are the nodes
% a triangle in counterclockwise order
% dirichlet: Matrix of size (n° of dirichlet nodes)x(2) where every row is
% a dirichlet node and its value (only zero values are supported)
% area: Area of the domain
% f: Function handle of the source heating function
% OUTPUT
% F: Discretization of the term <xi f>. The size of F is mx1

[ID,IEN,LM] = fe_locator(nodes,elements,dirichlet,1);
n_el=size(IEN,2);
m = max(ID);


nP = 9; % I'll over integrate because I don't know the form of f
ord_xi = 1;
[xi2D, w2D] = fe_GaussLeg2DTri(nP);
[xi,gradxixi] = fe_shapeTrivec(xi2D, ord_xi);


F = zeros(m,1);
n_en = 3; % order 1
for e=1:n_el
    xnod=nodes(IEN(:,e),:);
    fe = fe_localElemF(xnod,xi,gradxixi,f,w2D);
    for i=1:n_en
        if LM(i,e)~=0
            F(LM(i,e),1) = F(LM(i,e),1) + fe(i);
        end
    end
end
F = (1/area)*F;
end

