// Example adapted from: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/benchmarks/levelset/carreTri.geo
// and https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/benchmarks/levelset/test.geo
// 
// Generate a submesh on a circular domain from a mesh on the unit square and a given levelset function.
// "inside.msh" is the inside of the circular domain, "outside.msh" is the outside i.e. the unit square with a hole.

Plugin(CutMesh).SaveTri = 1;

l=2;
nb=10;
cl=l/nb;
half_pi=1.57079632679;

Point(1) = {0,       0,       0, cl};
Point(2) = {half_pi, 0,       0, cl};
Point(3) = {half_pi, half_pi, 0, cl};
Point(4) = {0,       half_pi, 0, cl};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Physical Line("Bottom") = {1};
Physical Line("Left")   = {2};
Physical Line("Top")    = {3};
Physical Line("Right")  = {4};

Line Loop(5) = {1,2,3,4};
Plane Surface(6) = {5};

Physical Surface(500) = {6};

Mesh 2;

Save "square.msh";
Delete Physicals;

Levelset MathEval (10) = "sqrt((x-0.5)^2 + (y-0.5)^2) - 0.3";
Levelset CutMesh {10};

// Outside
Physical Surface(2000) = {7};

Save "outside.msh";
Delete Physicals;

// Inside
Physical Surface(1000) = {6};
Save "inside.msh";