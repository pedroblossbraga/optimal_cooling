function [xi2DQ,w2DQ] = fe_GaussLeg2DQuad(n1D)
% Gauss-Legendre quadrature points and weights for a quad with nodes (-1,1)
% (-1,1) (1,-1) and (1,1)
% 
% INPUT
% n1D: Input for mylegpts. See mylegpts for more info
% OUTPUT
% xi2DQ: Gauss-Legendre quadrature points for a quad
% w2DQ: Gauss-Legendre quadrature weights for a quad

[x, w] = fe_mylegpts(n1D);

xi2DQ = zeros(n1D^2,2); w2DQ = zeros(1,n1D^2);

for i=1:n1D
    for j=1:n1D
        xi2DQ((i-1)*n1D + j,:) = [x(i), x(j)];
        w2DQ((i-1)*n1D + j) = w(i)*w(j);
    end
end

end

