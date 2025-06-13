function [K,Apt,Mtt] = fe_localElems(xnod,theta,gradxitheta,psi,w2D)
% Computes the contribution of the element composed by xnod
% INPUT
% xnod: Coordinates of the nodes that compose the element
% theta: Values of basis functions in each node corresponding to theta
% gradxitheta: Values of the gradient of each basis function in the nodes
% psi: Values of basis functions in each node corresponding to theta
% w2D: Gauss-Legendre weights for quadrature
% OUTPUT
% K: Contribution of the element to the stiffness matrix
% Apt: Cell array with the contributions of the element to the Ak matrices
% Mtt: Contribution of the element to the mass matrix

dof_theta = size(theta,2);
dof_psi = size(psi,2);

Mtt = zeros(dof_theta,dof_theta);
K = zeros(dof_theta,dof_theta);
Apt = zeros(dof_psi,dof_theta,3); 

for i = 1:size(theta,1)
    [~,detJ,gradxtheta] = fe_shapeTriElem(xnod,theta(i,:),gradxitheta);
    gradxpsi = gradxtheta;
    gradxxi = gradxtheta;
    gradxtpsi = [gradxpsi(2,:);
                -gradxpsi(1,:)];
    Mtt = Mtt + theta(i,:)'*theta(i,:)*detJ*w2D(i);
    K = K + gradxtheta'*gradxtheta*detJ*w2D(i);
    temp = gradxtpsi'*gradxxi;
    Apt(:,:,1) = Apt(:,:,1) + temp(:,1)*theta(i,:)*detJ*w2D(i);
    Apt(:,:,2) = Apt(:,:,2) + temp(:,2)*theta(i,:)*detJ*w2D(i);
    Apt(:,:,3) = Apt(:,:,3) + temp(:,3)*theta(i,:)*detJ*w2D(i);
end
end

