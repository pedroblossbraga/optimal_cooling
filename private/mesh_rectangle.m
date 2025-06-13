function [nodes,elements,dirichlet,area] = mesh_rectangle(beta, N1, N2)
% Constructs a rectangle mesh.
% INPUT
% beta: Parameter that controls the ratio of lengths of sides. beta = 1
% gives a square of side length 1. Most of the script is done for beta = 1
% and might not work or look bad if that's not the case.
% N1,N2: Amount of elements per side.
% OUTPUT
% nodes: Matrix of size (n° of nodes)x(2) where every row is the coordinate
% of a node
% elements: Matrix of size (n° of elements)x3 where every row are the nodes
% a triangle in counterclockwise order
% dirichlet: Matrix of size (n° of dirichlet nodes)x(2) where every row is
% a dirichlet node and its value (only zero values are supported)
% area: Area of the domain

a = (1+beta)/(2*beta); b = (1+beta)/2;
a_b = -a/2; a_t = a/2; 
b_b = -b/2; b_t = b/2;
h1 = a/N1; h2 = b/N2;
area = a*b;
[X,Y]=meshgrid(a_b:h1:a_t, b_b:h2:b_t);
X=X(:); Y=Y(:); XY=[X,Y];
%Remove nodes outside the domain: 0.5*hr away from boundary
tol=4*eps;
XYIC=XY(((abs(XY(:,1))<a_t-tol) & (abs(XY(:,2))<b_t-tol)),:);
%Boundary
XYBD=XY(((abs(XY(:,1))>=a_t-tol) | (abs(XY(:,2))>=b_t-tol)),:);
nodes=[XYIC;XYBD];
elements=delaunay(nodes(:,1),nodes(:,2)); 
nodalbdryind=size(XYIC,1)+1:size(nodes,1); %only last nodes are bdry
dirichlet=[nodalbdryind(:),zeros(length(nodalbdryind),1)];
end