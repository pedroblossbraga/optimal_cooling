function plot_mesh(nodes,elements,dirichlet)
% Plots the mesh, with red circle to indicate the dirchlet nodes
% INPUT
% nodes: Matrix of size (n° of nodes)x(2) where every row is the coordinate
% of a node
% elements: Matrix of size (n° of elements)x3 where every row are the nodes
% a triangle in counterclockwise order
% dirichlet: Matrix of size (n° of dirichlet nodes)x(2) where every row is
% a dirichlet node and its value (only zero values are supported)

figure
hold on
triplot(elements(:,1:3),nodes(:,1),nodes(:,2))
plot(nodes(dirichlet(:,1),1),nodes(dirichlet(:,1),2),'ro')
axis equal
hold off

end

