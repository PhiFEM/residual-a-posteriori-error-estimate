l=1.;
nb=10;
lc=l/nb;

Point(1) = {0.,  -0.5, 0., lc};
Point(2) = {0.5, -0.5, 0., lc};
Point(3) = {0.5,  0.5, 0., lc};
Point(4) = {-0.5, 0.5, 0., lc};
Point(5) = {-0.5,  0., 0., lc};
Point(6) = {0.,    0., 0., lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Line Loop(5) = {1, 2, 3, 4, 5, 6};
Plane Surface(6) = {5};

Physical Surface(100) = {6};

Mesh 2;

Save "lshaped.msh";