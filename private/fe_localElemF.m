function fe = fe_localElemF(xnod,xi,gradxixi,f,w2D)
% Computes the contribution of the element composed by xnod to the vector F
% INPUT
% xnod: Coordinates of the nodes that compose the element
% xi: Values of basis functions in each node corresponding to xi
% gradxixi: Values of the gradient of each basis function in the nodes
% f: Function handle of the source heating function
% w2D: Gauss-Legendre weights for quadrature
% OUTPUT
% fe: Contribution of the element to the F vector

% [xi2D, w2D] = GaussLeg2DTri(nP);
% ord_xi = 1; dof_xi = ord_xi * 3;
dof_xi = size(xi,2);


fe = zeros(1,dof_xi);
for i = 1:size(xi,1)
    % xieta = xi2D(i,:);
    % [xi,gradxixi]=shapeTri(xieta,ord_xi);
    [xy,detJ,~] = fe_shapeTriElem(xnod,xi(i,:),gradxixi);
    fe = fe + xi(i,:)*f(xy(1),xy(2))*detJ*w2D(i);
end
end

