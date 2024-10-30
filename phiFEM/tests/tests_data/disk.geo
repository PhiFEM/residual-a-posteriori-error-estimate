lc = 0.25;

Point(1) = { 0.0,  0.0, 0.0, lc};
Point(2) = { 0.5,  0.0, 0.0, lc};
Point(3) = {-0.5,  0.0, 0.0, lc};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 2};

Line Loop(1) = {1, 2};
Plane Surface(1) = {1};

Mesh 2;