function [ID,IEN,LM] = fe_locator(nodes,elements,dirichlet,order)
%LOCATOR Constructs the ID,IEN,LM matrices with the structure explained by
% Hughes.
% INPUT
% nodes: Matrix of size (n° of nodes)x(2) where every row is the coordinate
% of a node
% elements: Matrix of size (n° of elements)x3 where every row are the nodes
% a triangle in counterclockwise order
% dirichlet: Matrix of size (n° of dirichlet nodes)x(2) where every row is
% a dirichlet node and its value (only zero values are supported)
% order: Order of the basis functions. It might not work as intented for
% order ~= 1.
% OUTPUT
% ID: Matrix of size 2x(number of nodes), where every column corresponds to
% the degree of freedom of a node
% IEN: Matrix of size 3x(number of elements), where every column
% corresponds to an element and the nodes that compose it
% LM: Matrix of size 3x(number of elements), where every column
% corresponds to an element and the degrees of freedom of the nodes that 
% compose it

n_en = 3*order;

% ID
i = 1:length(nodes); idx = ~ismember(i, dirichlet(:,1));
ID = sparse(1,i(idx),1:nnz(idx),1,length(nodes));

IEN = elements';

% LM
LM = zeros(n_en,length(elements));
for i = 1:length(elements)
    for j=1:n_en
        LM(j,i) = ID(IEN(j,i));
    end
end

end

