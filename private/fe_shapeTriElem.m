function [xy,detJ,gradxphi] = fe_shapeTriElem(xnod,phi,gradxiphi)
%SHAPETRIELEM Converts the coordinate and gradient in the master element to
% physical values.
% Optimized for order 1 and doesn't work for order 2
xy = phi*xnod;
mat = gradxiphi*xnod;
detJ = mat(1,1)*mat(2,2) - mat(1,2)*mat(2,1);
invmat = (1/detJ) * [mat(2,2), -mat(1,2);
                    -mat(2,1), mat(1,1)];
gradxphi = invmat*gradxiphi;
end

