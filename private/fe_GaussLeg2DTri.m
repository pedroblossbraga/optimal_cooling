function [xi2D, w2D] = fe_GaussLeg2DTri(nP)
% Gauss-Legendre quadrature points and weights for a triangle with nodes 
% (0,0) (1,0) and (0,1)
% INPUT
% nP: Input for mylegpts. See mylegpts for more info
% OUTPUT
% xi2D: Gauss-Legendre quadrature points for a triangle
% w2D: Gauss-Legendre quadrature weights for a triangle

[xi2DQ,w2DQ] = fe_GaussLeg2DQuad(nP);

w2D = w2DQ' * 0.25 .* (1 - 0.5 * (xi2DQ(:,2) + 1));
w2D = w2D';
xi2D = zeros(nP^2,2);
xi2D(:,1) = 0.5 * (xi2DQ(:,1) + 1) .* (1 - 0.5 * (xi2DQ(:,2) + 1));
xi2D(:,2) = 0.5 * (xi2DQ(:,2) + 1);


end

