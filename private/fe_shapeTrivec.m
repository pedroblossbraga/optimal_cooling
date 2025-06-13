function [phi,gradxiphi] = fe_shapeTrivec(xieta, order)
%SHAPETRI Evaluates shape functions and their derivatives on xieta = [vec(xi), vec(eta)]. 
% It's useful to define the barycentric coordinates: (s0,s1,s2)
% r = xi, s = eta, t = 1 - xi - eta
%   order = 1: phi1(r,s,t) = r -grad> [1;0] 
%              phi2(r,s,t) = s -grad> [0;1]
%              phi3(r,s,t) = t -grad> [-1;-1]
%   order = 2: phi4(r,s,t) = 4rs       -grad> [4s;4r]
%              phi5(r,s,t) = 4st       -grad> [-4s;4(t-s)]
%              phi6(r,s,t) = 4rt       -grad> [4(t-r);-4r]
% Barycentric coordinates
r = xieta(:,1); s = xieta(:,2); t = 1 - r - s;
if order == 1
    phi = [r, s, t];
    gradxiphi = [1, 0, -1; 
                 0, 1, -1];
elseif order == 2
    phi = [r, s, t, 4*r*s, 4*s*t, 4*r*t];
    gradxiphi = [1, 0, -1, 4*s, -4*s, 4*(t - r);
                 0, 1, -1, 4*r, 4*(t - s), -4*r];
    % phi = [r*(2*r - 1), s*(2*s - 1), t*(2*t - 1), 4*r*s, 4*s*t, 4*r*t];
    % gradxiphi = [4*r - 1, 0, -(4*t - 1), 4*s, -4*s, 4*(t - r);
    %              0, 4*s - 1, -(4*t - 1), 4*r, 4*(t - s), -4*r];
end
end